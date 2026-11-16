#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import zipfile

import numpy as np

IMAGE_SIZE = 640
NETWORK_NAME = "bird_yolo"
STRIDES = (8, 16, 32)
NMS_SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.7
MAX_BOXES = 100
REGRESSION_LENGTH = 16
CHECK_IMAGES = 16
GRAY_STACK_INPUT_NAME = "grays"
HEAD_CONV = re.compile(r"^/model\.(\d+)/cv([23])\.(\d+)/cv[23]\.\d+\.2/Conv$")


def export_onnx(weights_path):
    from ultralytics import YOLO
    return YOLO(weights_path).export(format="onnx", imgsz=IMAGE_SIZE, opset=11, simplify=True, dynamic=False)


def detect_head_outputs(onnx_path):
    import onnx
    graph = onnx.load(onnx_path).graph
    matches = [(int(m.group(1)), m.group(2), int(m.group(3)), node.name)
               for node in graph.node for m in [HEAD_CONV.match(node.name)] if m]
    if not matches:
        raise SystemExit("could not find the detection head convolutions (/model.N/cv2.i/cv2.i.2/Conv); is this a YOLOv8/YOLO11 export?")
    head_index = max(index for index, _, _, _ in matches)
    head = {(branch, scale): name for index, branch, scale, name in matches if index == head_index}
    box_nodes = [head[("2", scale)] for scale in range(len(STRIDES))]
    class_nodes = [head[("3", scale)] for scale in range(len(STRIDES))]
    return box_nodes, class_nodes, graph.input[0].name


def layers_by_original_name(runner):
    hn = runner.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    names = {}
    for layer_name, layer in hn["layers"].items():
        for original in layer.get("original_names", []):
            names[original] = layer_name
    return names


def short_name(layer_name):
    return layer_name.split("/")[-1]


def write_nms_config(box_layers, class_layers, path):
    decoders = [{"name": f"{NETWORK_NAME}/bbox_decoder_{scale}", "stride": stride,
                 "reg_layer": box_layers[scale], "cls_layer": class_layers[scale]}
                for scale, stride in enumerate(STRIDES)]
    config = {"nms_scores_th": NMS_SCORE_THRESHOLD, "nms_iou_th": NMS_IOU_THRESHOLD, "image_dims": [IMAGE_SIZE, IMAGE_SIZE],
              "max_proposals_per_class": MAX_BOXES, "classes": 1, "regression_length": REGRESSION_LENGTH,
              "background_removal": False, "background_removal_index": 0, "bbox_decoders": decoders}
    with open(path, "w") as f:
        json.dump(config, f, indent=1)


def has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def model_script(class_layers, nms_config_path, input_name, nms_engine, noise_analysis=False):
    lines = [] if input_name == GRAY_STACK_INPUT_NAME else ["normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])"]
    lines += [f"change_output_activation({short_name(layer)}, sigmoid)" for layer in class_layers]
    lines.append(f"model_optimization_flavor(optimization_level={2 if has_gpu() else 1}, compression_level=0)")
    if not noise_analysis:
        lines.append("model_optimization_config(checker_cfg, policy=disabled)")
    lines.append(f'nms_postprocess("{nms_config_path}", meta_arch=yolov8, engine={nms_engine})')
    return "\n".join(lines) + "\n"


def calibration_images(path, count=None):
    import tensorflow as tf
    folder = path
    if path.endswith(".zip"):
        folder = path[:-4]
        with zipfile.ZipFile(path) as archive:
            archive.extractall(folder)
    files = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
    if not files:
        raise SystemExit(f"no .jpg or .png calibration crops in {path}")
    if count:
        files = files[::max(1, len(files) // count)][:count]
    images = np.empty((len(files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for index, file in enumerate(files):
        image = tf.io.decode_image(tf.io.read_file(file), channels=3, expand_animations=False).numpy()
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise SystemExit(f"calibration crops must be {IMAGE_SIZE}x{IMAGE_SIZE}: {file} is {image.shape}")
        images[index] = image
    return images


def boxes_over_threshold(output):
    scores = np.asarray(output, dtype=np.float32)
    box_axis = next((axis for axis in range(1, scores.ndim) if scores.shape[axis] == 5), None)
    if box_axis is None:
        return None, scores.shape
    scores = np.moveaxis(scores, box_axis, -1)[..., 4]
    return (scores >= NMS_SCORE_THRESHOLD).reshape(len(scores), -1).sum(axis=1), scores.shape


def compare_float_and_int8(runner, images):
    from hailo_sdk_client import InferenceContext
    results = {}
    for name, context in (("float", InferenceContext.SDK_FP_OPTIMIZED), ("int8", InferenceContext.SDK_QUANTIZED)):
        with runner.infer_context(context) as ctx:
            results[name] = boxes_over_threshold(runner.infer(ctx, images))
    for name, (counts, shape) in results.items():
        if counts is None:
            print(f"{name}: emulator output shape {shape}; could not count boxes automatically")
        else:
            print(f"{name}: birds found in {int((counts > 0).sum())}/{len(counts)} check crops, {int(counts.sum())} boxes in total")


def compile_to_hef(runner, output):
    hef = runner.compile()
    with open(output, "wb") as f:
        f.write(hef)
    runner.save_har(output[:-4] + "_compiled.har")
    print("saved", output)


def main():
    parser = argparse.ArgumentParser(description="Compile the bird YOLO (YOLOv8/YOLO11) into a Hailo .hef, quantized to INT8 with motion-view crops.")
    parser.add_argument("--weights", help="bird_yolo11n_v1.pt (exported to ONNX at 640 first)")
    parser.add_argument("--onnx", help="an existing 640x640 ONNX export instead of --weights")
    parser.add_argument("--calibration", default="calibration_crops.zip")
    parser.add_argument("--calibration-count", type=int, help="use only this many crops, spread over the set, to save memory")
    parser.add_argument("--hw-arch", default="hailo10h", help="hailo10h for the AI HAT+ 2")
    parser.add_argument("--output", default="bird_yolo11n_v1.hef")
    parser.add_argument("--resume", action="store_true", help="skip to compiling the optimized model saved by an earlier run")
    parser.add_argument("--nms-engine", choices=["cpu", "nn_core", "auto"], default="cpu",
                        help="where duplicate-box removal runs: cpu = HailoRT on the Pi CPU, nn_core = on the Hailo chip")
    parser.add_argument("--noise-analysis", action="store_true",
                        help="run Hailo's per-layer noise report after optimizing (memory hungry; off by default)")
    parser.add_argument("--optimize-only", action="store_true", help="stop after saving the optimized model, so compiling can run in a fresh process with more free memory")
    args = parser.parse_args()
    os.environ.setdefault("USER", "colab")
    optimized_har = args.output[:-4] + "_optimized.har"

    from hailo_sdk_client import ClientRunner
    if args.resume:
        runner = ClientRunner(har=optimized_har, hw_arch=args.hw_arch)
        compile_to_hef(runner, args.output)
        return
    if not (args.weights or args.onnx):
        raise SystemExit("give --weights or --onnx")
    onnx_path = args.onnx or export_onnx(args.weights)
    box_nodes, class_nodes, input_name = detect_head_outputs(onnx_path)
    print("input:", input_name, "| head outputs:", box_nodes + class_nodes)

    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(onnx_path, NETWORK_NAME, start_node_names=[input_name], end_node_names=box_nodes + class_nodes,
                                net_input_shapes={input_name: [1, 3, IMAGE_SIZE, IMAGE_SIZE]})
    names = layers_by_original_name(runner)
    missing = [node for node in box_nodes + class_nodes if node not in names]
    if missing:
        raise SystemExit(f"parsed network has no layers for {missing}; layers found: {sorted(set(names.values()))}")
    box_layers = [names[node] for node in box_nodes]
    class_layers = [names[node] for node in class_nodes]
    nms_config_path = os.path.abspath("bird_nms_config.json")
    write_nms_config(box_layers, class_layers, nms_config_path)
    script = model_script(class_layers, nms_config_path, input_name, args.nms_engine, args.noise_analysis)
    print("model script:\n" + script)
    runner.load_model_script(script)

    images = calibration_images(args.calibration, args.calibration_count)
    print(f"optimizing with {len(images)} calibration crops (GPU: {has_gpu()})")
    runner.optimize(images)
    runner.save_har(optimized_har)
    print("saved", optimized_har, "(re-run with --resume to compile from here)")
    compare_float_and_int8(runner, images[:CHECK_IMAGES])
    if args.optimize_only:
        return
    del images
    compile_to_hef(runner, args.output)


if __name__ == "__main__":
    main()
