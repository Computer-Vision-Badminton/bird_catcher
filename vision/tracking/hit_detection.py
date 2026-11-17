import collections
import time

import numpy as np

from vision.tracking.bird_tracker import next_rising_step
from vision.detection.crop_tracking import CROP_SIZE, FOLLOW_CROP_SIZE, CropFollower, crop_origin, right_crop_center
from vision.camera.frames import crop_eye, plain_gray
from vision.pipeline.shot_pipeline import BlurredFrameCache, blobs_near

FRAMES_PER_VIEW = 3
FRAMES_TO_SEARCH_NEAR_HEADING = 10
CHAIN_LOOKBACK_FRAMES = 40
REFRACTORY_FRAMES = 40
HIT_RULE = {"min_points": 3, "min_rise_px": 20, "min_detection_confidence": 0.25, "max_span_frames": 20,
            "min_start_row": 350, "estimate": "first-point", "offset": -1}


def milliseconds_since(started):
    return (time.perf_counter() - started) * 1000


def detections_in_image(detections_in_crop, crop_box):
    left, top, _ = crop_box
    return [(x + left, y + top, confidence) for x, y, confidence in detections_in_crop]


def rounded_detections(detections):
    return [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in detections]


class WaitingDetector:
    def __init__(self, search_detector, follow_detector, overlap=False, right_row_offset=None):
        self.search_detector = search_detector
        self.follow_detector = follow_detector
        self.overlap = overlap
        self.right_row_offset = right_row_offset
        self.reset()

    def follows_with_both_eyes(self):
        return self.right_row_offset is not None

    def reset(self):
        self.grays = collections.deque(maxlen=FRAMES_PER_VIEW)
        self.follower = CropFollower()
        self.blurred_frames = BlurredFrameCache()
        self.pending = None
        self.last_mode = None
        self.stereo_frames = collections.deque(maxlen=FRAMES_PER_VIEW)
        self.right_grays = collections.deque(maxlen=FRAMES_PER_VIEW)

    def right_eye_grays(self):
        for position, gray in enumerate(self.right_grays):
            if gray is None:
                self.right_grays[position] = plain_gray(crop_eye(self.stereo_frames[position], "right"))
        return list(self.right_grays)

    def heading_still_useful(self, frame):
        return bool(self.follower.recent) and frame - self.follower.recent[-1][0] <= FRAMES_TO_SEARCH_NEAR_HEADING

    def aim(self, frame, speculative_blobs=None):
        height, width = self.grays[-1].shape[:2]
        if not self.follower.needs_blobs(frame):
            return "follow", self.follower.crop_center(frame, [], width, height), FOLLOW_CROP_SIZE
        if self.heading_still_useful(frame):
            heading = self.follower.last_seen_heading(frame)
            blobs = blobs_near(self.grays, heading)
            return "near", self.follower.crop_center(frame, blobs, width, height, without_blobs=heading), FOLLOW_CROP_SIZE
        self.follower.recent = []
        blobs = self.blurred_frames.blobs_around(self.grays) if speculative_blobs is None else speculative_blobs
        return "search", self.follower.crop_center(frame, blobs, width, height), CROP_SIZE

    def push(self, stereo_frame):
        started = time.perf_counter()
        self.grays.append(plain_gray(crop_eye(stereo_frame, "left")))
        if self.follows_with_both_eyes():
            self.stereo_frames.append(stereo_frame)
            self.right_grays.append(None)
        return milliseconds_since(started), len(self.grays) == FRAMES_PER_VIEW

    def cut_crop(self, center, crop_size, grays=None):
        grays = self.grays if grays is None else grays
        height, width = grays[-1].shape[:2]
        left, top = crop_origin(*center, width, height, crop_size)
        crop = np.ascontiguousarray(np.dstack([gray[top:top + crop_size, left:left + crop_size] for gray in grays]))
        detector = self.search_detector if crop_size == CROP_SIZE else self.follow_detector
        return (left, top, crop_size), crop, detector

    def cut_crops(self, center, crop_size):
        crop_box, crop, detector = self.cut_crop(center, crop_size)
        if not self.follows_with_both_eyes() or crop_size != FOLLOW_CROP_SIZE:
            return [crop_box], [crop], detector
        right_box, right_crop, _ = self.cut_crop(right_crop_center(center, self.right_row_offset), crop_size, self.right_eye_grays())
        return [crop_box, right_box], [crop, right_crop], detector

    def detect_middle(self, frame):
        started = time.perf_counter()
        mode, center, crop_size = self.aim(frame)
        aimed = time.perf_counter()
        crop_boxes, crops, detector = self.cut_crops(center, crop_size)
        cut = time.perf_counter()
        detections_per_crop = [detector.detect(crops[0])] if len(crops) == 1 else detector.detect_batch(crops)
        detected = time.perf_counter()
        return self.remembered_entry(frame, mode, crop_boxes, detections_per_crop, (aimed - started) * 1000,
                                     (detected - aimed) * 1000, (detected - cut) * 1000)

    def remembered_entry(self, frame, mode, crop_boxes, detections_per_crop, aim_ms, yolo_ms, detector_call_ms):
        detections, *right_eye = [detections_in_image(detections_in_crop, crop_box)
                                  for detections_in_crop, crop_box in zip(detections_per_crop, crop_boxes)]
        self.follower.remember(frame, detections)
        left, top, crop_size = crop_boxes[0]
        entry = {"mode": mode, "crop": [int(left), int(top), int(crop_size)], "detections": rounded_detections(detections),
                 "aim_ms": round(aim_ms, 2), "yolo_ms": round(yolo_ms, 2), "detector_call_ms": round(float(detector_call_ms), 2)}
        if right_eye:
            right_left, right_top, _ = crop_boxes[1]
            entry.update({"right_crop": [int(right_left), int(right_top), int(crop_size)], "right_detections": rounded_detections(right_eye[0]),
                          "stereo_detections": (detections, right_eye[0])})
        return entry

    def speculative_blobs(self):
        if self.last_mode != "search":
            return None, 0.0
        started = time.perf_counter()
        blobs = self.blurred_frames.blobs_around(self.grays)
        return blobs, milliseconds_since(started)

    def finish_pending(self):
        if self.pending is None:
            return None, 0.0
        frame, mode, crop_boxes, aim_ms, start_ms, job = self.pending
        self.pending = None
        started = time.perf_counter()
        detections_per_crop = job.wait()
        wait_ms = milliseconds_since(started)
        entry = self.remembered_entry(frame, mode, crop_boxes, detections_per_crop, aim_ms, start_ms, job.call_ms)
        return (frame, entry), wait_ms

    def start_middle(self, frame, speculative_blobs, speculative_blobs_ms, wait_ms):
        started = time.perf_counter()
        mode, center, crop_size = self.aim(frame, speculative_blobs)
        aimed = time.perf_counter()
        crop_boxes, crops, detector = self.cut_crops(center, crop_size)
        job = detector.start_batch(crops)
        start_ms = (time.perf_counter() - aimed) * 1000
        aim_ms = speculative_blobs_ms + (aimed - started) * 1000
        self.pending = (frame, mode, crop_boxes, aim_ms, start_ms + wait_ms, job)
        self.last_mode = mode

    def step(self, stereo_frame, newest_frame):
        gray_ms, ready = self.push(stereo_frame)
        middle = newest_frame - 1
        started_middle = middle if ready else None
        if not self.overlap:
            return gray_ms, started_middle, (middle, self.detect_middle(middle)) if ready else None
        speculative_blobs, speculative_blobs_ms = self.speculative_blobs() if ready else (None, 0.0)
        finished, wait_ms = self.finish_pending()
        if ready:
            self.start_middle(middle, speculative_blobs, speculative_blobs_ms, wait_ms)
        return gray_ms, started_middle, finished

    def drain(self):
        return self.finish_pending()[0]

    def pending_stereo_frame(self):
        if self.pending is None or len(self.pending[2]) != 2:
            return None
        return self.pending[0]

    def hand_over_pending(self):
        frame, _, crop_boxes, _, _, job = self.pending
        self.pending = None
        return frame, crop_boxes[0][2], [(left, top) for left, top, _ in crop_boxes], job


def chain_from(found, start_frame, x, y):
    chain = [(start_frame, x, y)]
    while True:
        step = next_rising_step(found, *chain[-1])
        if step is None:
            return chain
        chain.append(step)


def best_chain_ending_at(found, newest_frame):
    visible = {frame: found[frame] for frame in range(newest_frame - CHAIN_LOOKBACK_FRAMES, newest_frame + 1) if frame in found}
    best = []
    for start_frame in sorted(visible):
        for x, y, _ in visible[start_frame]:
            chain = chain_from(visible, start_frame, x, y)
            if chain[-1][0] == newest_frame and len(chain) > len(best):
                best = chain
    return best


def hit_estimate(chain, offset=HIT_RULE["offset"]):
    return chain[0][0] - offset


def chain_fires(chain, rule=HIT_RULE):
    if len(chain) < rule["min_points"] or chain[0][2] - chain[-1][2] < rule["min_rise_px"]:
        return False
    if chain[-1][0] - chain[0][0] > rule["max_span_frames"]:
        return False
    return chain[0][2] >= rule["min_start_row"]


def hit_event(frame, chain, estimate):
    return {"fired_at": frame + 1, "chain_first_frame": chain[0][0], "chain": [list(point) for point in chain],
            "hit_estimate": estimate}


class HitWatcher:
    def __init__(self, rule=HIT_RULE):
        self.rule = rule
        self.reset()

    def reset(self):
        self.found = {}
        self.quiet_until = -1

    def forget_before(self, frame):
        for old_frame in [old_frame for old_frame in self.found if old_frame < frame]:
            del self.found[old_frame]

    def push(self, frame, detections):
        confident = [(x, y, c) for x, y, c in detections if c >= self.rule["min_detection_confidence"]]
        self.found[frame] = confident
        self.forget_before(frame - CHAIN_LOOKBACK_FRAMES)
        if not confident or frame <= self.quiet_until:
            return None
        chain = best_chain_ending_at(self.found, frame)
        if len(chain) < 2 or not chain_fires(chain, self.rule):
            return None
        self.quiet_until = frame + REFRACTORY_FRAMES
        return hit_event(frame, chain, hit_estimate(chain, self.rule["offset"]))
