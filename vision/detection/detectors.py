import concurrent.futures
import contextlib
import time

import cv2
import numpy as np

from vision.detection.crop_tracking import CROP_SIZE, MIN_CONFIDENCE

DETECTOR_KINDS = ("ncnn", "pt", "hailo", "onnx")
MOTION_VIEW_INPUT = "motion-view"
GRAY_STACK_INPUT = "gray-stack"
INPUT_KINDS = (MOTION_VIEW_INPUT, GRAY_STACK_INPUT)
NMS_IOU = 0.7
END_TO_END_BOX_VALUES = 6
NMS_BOX_VALUES = 5
HAILO_TIMEOUT_MS = 1000
STEREO_BATCH = 2


def milliseconds_since(started):
    return (time.perf_counter() - started) * 1000


def timed_detect_batch(detect_batch, crops):
    started = time.perf_counter()
    detections = detect_batch(crops)
    return detections, milliseconds_since(started)


class ThreadDetectionJob:
    def __init__(self, future):
        self.future = future
        self.call_ms = None

    def wait(self):
        detections, self.call_ms = self.future.result()
        return detections


class FinishedDetectionJob:
    def __init__(self, detections, call_ms):
        self.detections = detections
        self.call_ms = call_ms

    def wait(self):
        return self.detections


class StartsBatchesOnAWorkerThread:
    detection_worker = None

    def start_batch(self, crops):
        if self.detection_worker is None:
            self.detection_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return ThreadDetectionJob(self.detection_worker.submit(timed_detect_batch, self.detect_batch, list(crops)))


class UltralyticsDetector(StartsBatchesOnAWorkerThread):
    input_kind = MOTION_VIEW_INPUT

    def __init__(self, model_path, device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="detect")
        self.device = device if model_path.endswith(".pt") else None

    def detect(self, crop):
        result = self.model.predict(crop, imgsz=CROP_SIZE, conf=MIN_CONFIDENCE, device=self.device, verbose=False)[0]
        centers = result.boxes.xywh.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        return [(float(x), float(y), float(c)) for (x, y, _, _), c in zip(centers, confidences)]

    def detect_batch(self, crops):
        return [self.detect(crop) for crop in crops]


class OnnxDetector(StartsBatchesOnAWorkerThread):
    def __init__(self, model_path, input_kind=MOTION_VIEW_INPUT):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_kind = input_kind

    def detect(self, crop):
        image = crop if self.input_kind == GRAY_STACK_INPUT else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        scale = 1.0 if self.input_kind == GRAY_STACK_INPUT else 1 / 255
        batch = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) * scale
        rows = self.session.run(None, {self.input_name: batch})[0][0].T
        rows = rows[rows[:, 4] >= MIN_CONFIDENCE]
        boxes = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h, _ in rows]
        kept = cv2.dnn.NMSBoxes(boxes, rows[:, 4].tolist(), MIN_CONFIDENCE, NMS_IOU) if len(rows) else []
        return [(float(rows[i, 0]), float(rows[i, 1]), float(rows[i, 4])) for i in np.asarray(kept).reshape(-1)]

    def detect_batch(self, crops):
        return [self.detect(crop) for crop in crops]


def describe_output(value):
    if isinstance(value, np.ndarray):
        return f"array{value.shape}"
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}] of " + (describe_output(value[0]) if value else "nothing")
    return type(value).__name__


def boxes_from_nms_output(per_class_boxes, input_width, input_height):
    detections = []
    for class_boxes in per_class_boxes:
        for y_min, x_min, y_max, x_max, score in np.asarray(class_boxes).reshape(-1, NMS_BOX_VALUES):
            if score >= MIN_CONFIDENCE:
                detections.append(((x_min + x_max) / 2 * input_width, (y_min + y_max) / 2 * input_height, float(score)))
    return detections


def boxes_from_end_to_end_output(rows, input_width, input_height):
    rows = np.asarray(rows).reshape(-1, END_TO_END_BOX_VALUES)
    scale = (input_width, input_height) if rows.size and np.abs(rows[:, :4]).max() <= 1.5 else (1.0, 1.0)
    return [((x1 + x2) / 2 * scale[0], (y1 + y2) / 2 * scale[1], float(score))
            for x1, y1, x2, y2, score, _ in rows if score >= MIN_CONFIDENCE]


def copied_output(value):
    if isinstance(value, (list, tuple)):
        return [copied_output(part) for part in value]
    return np.array(value, copy=True)


class HailoDetectionJob:
    def __init__(self, detector, crops, inputs):
        self.detector = detector
        self.crops = crops
        self.inputs = inputs
        self.started = time.perf_counter()
        self.call_ms = None
        self.failure = None
        self.async_job = None

    def on_done(self, completion_info):
        self.call_ms = milliseconds_since(self.started)
        self.failure = getattr(completion_info, "exception", None)

    def wait(self):
        self.async_job.wait(HAILO_TIMEOUT_MS)
        self.detector.running_job = None
        if self.failure:
            raise RuntimeError(f"Hailo asynchronous run failed: {self.failure}")
        if self.call_ms is None:
            self.call_ms = milliseconds_since(self.started)
        return self.detector.detect_from_bindings(self.crops)


DEVICES_THAT_HAVE_RUN_INFERENCE = set()


class HailoDetector:
    def __init__(self, hef_path, input_is_bgr=False, input_kind=MOTION_VIEW_INPUT, batch_size=1, shared_device=None):
        from hailo_platform import FormatType, VDevice
        self.input_is_bgr = input_is_bgr
        self.input_kind = input_kind
        self.resources = contextlib.ExitStack()
        if shared_device is not None and id(shared_device) in DEVICES_THAT_HAVE_RUN_INFERENCE:
            raise RuntimeError("create every Hailo model on a shared VDevice before the first inference; "
                               "create_infer_model times out once the device has run")
        self.device = shared_device or self.resources.enter_context(VDevice())
        self.model = self.device.create_infer_model(hef_path)
        self.batch_size = batch_size
        self.model.set_batch_size(batch_size)
        self.model.input().set_format_type(FormatType.UINT8)
        for output in self.model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        self.input_height, self.input_width = self.model.input().shape[:2]
        self.output_name = self.model.output().name
        self.configured = self.resources.enter_context(self.model.configure())
        self.bindings = [self.configured.create_bindings(
            output_buffers={self.output_name: np.empty(self.model.output().shape, dtype=np.float32)})
            for _ in range(batch_size)]
        self.running_job = None

    def describe(self):
        return {"input": {"name": self.model.input().name, "shape": list(self.model.input().shape)},
                "outputs": [{"name": output.name, "shape": list(output.shape)} for output in self.model.outputs]}

    def prepared_input(self, crop):
        image = crop if self.input_is_bgr or self.input_kind == GRAY_STACK_INPUT else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.input_height, self.input_width):
            image = cv2.resize(image, (self.input_width, self.input_height))
        return np.ascontiguousarray(image, dtype=np.uint8)

    def raw_outputs(self, crops):
        outputs = []
        for start in range(0, len(crops), self.batch_size):
            outputs += self.raw_outputs_of_one_call(crops[start:start + self.batch_size])
        return outputs

    def fill_bindings(self, crops):
        if self.running_job is not None:
            raise RuntimeError("wait for the running Hailo job before starting another on the same bindings")
        padded = (list(crops) + [crops[-1]] * self.batch_size)[:self.batch_size]
        inputs = [self.prepared_input(crop) for crop in padded]
        for binding, image in zip(self.bindings, inputs):
            binding.input().set_buffer(image)
        return inputs

    def outputs_from_bindings(self, crop_count):
        return [copied_output(binding.output(self.output_name).get_buffer()) for binding in self.bindings[:crop_count]]

    def raw_outputs_of_one_call(self, crops):
        self.fill_bindings(crops)
        DEVICES_THAT_HAVE_RUN_INFERENCE.add(id(self.device))
        self.configured.run(self.bindings, HAILO_TIMEOUT_MS)
        return self.outputs_from_bindings(len(crops))

    def start_batch(self, crops):
        crops = list(crops)
        if len(crops) > self.batch_size:
            started = time.perf_counter()
            return FinishedDetectionJob(self.detect_batch(crops), milliseconds_since(started))
        inputs = self.fill_bindings(crops)
        job = HailoDetectionJob(self, crops, inputs)
        DEVICES_THAT_HAVE_RUN_INFERENCE.add(id(self.device))
        self.configured.wait_for_async_ready(timeout_ms=HAILO_TIMEOUT_MS)
        job.async_job = self.configured.run_async(self.bindings, job.on_done)
        self.running_job = job
        return job

    def detect_from_bindings(self, crops):
        return [self.detections_in_crop(output, crop) for output, crop in zip(self.outputs_from_bindings(len(crops)), crops)]

    def raw_output(self, crop):
        return self.raw_outputs([crop])[0]

    def detect(self, crop):
        return self.detect_batch([crop])[0]

    def detect_batch(self, crops):
        return [self.detections_in_crop(output, crop) for output, crop in zip(self.raw_outputs(crops), crops)]

    def detections_in_crop(self, output, crop):
        scale_x, scale_y = crop.shape[1] / self.input_width, crop.shape[0] / self.input_height
        if isinstance(output, (list, tuple)):
            detections = boxes_from_nms_output(output, self.input_width, self.input_height)
        elif isinstance(output, np.ndarray) and output.shape[-1] == END_TO_END_BOX_VALUES:
            detections = boxes_from_end_to_end_output(output, self.input_width, self.input_height)
        else:
            raise RuntimeError(f"unrecognised Hailo output {self.output_name}: {describe_output(output)}; "
                               "expected a per-class list of NMS boxes or end-to-end rows of (x1, y1, x2, y2, score, class)")
        return [(x * scale_x, y * scale_y, confidence) for x, y, confidence in detections]

    def close(self):
        self.resources.close()


def load_detector(kind, model_path, hailo_input_is_bgr=False, input_kind=MOTION_VIEW_INPUT, hailo_batch_size=STEREO_BATCH,
                  on_hailo_device_of=None):
    if kind == "hailo":
        shared_device = on_hailo_device_of.device if on_hailo_device_of is not None else None
        return HailoDetector(model_path, hailo_input_is_bgr, input_kind, hailo_batch_size, shared_device)
    if kind == "onnx":
        return OnnxDetector(model_path, input_kind)
    if input_kind != MOTION_VIEW_INPUT:
        raise SystemExit(f"{kind} models take the motion view; use --detector onnx or hailo for {input_kind}")
    return UltralyticsDetector(model_path)
