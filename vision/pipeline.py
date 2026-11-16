import collections
import time

import cv2
import numpy as np

from bird_tracker import EYES, FRAMES_AFTER_HIT, longest_rising_chain, match_in_right_eye, moving_blobs
from crop_tracking import CROP_SIZE, CropFollower, crop_origin, right_eye_aim
from detectors import GRAY_STACK_INPUT
from flight_model import flight_points
from frames import crop_eye, plain_gray
from landing_worker import LandingPredictor
from motion_view import current_position_motion_view

STAGES = ("gray", "motion_view", "aim_left", "yolo_left", "aim_right", "yolo_right", "track", "predict")
FRAMES_NEEDED_PER_VIEW = 3
FIT_EVERY_FRAMES = 1


class StageClock:
    def __init__(self):
        self.timings = {}
        self.started = time.perf_counter()

    def lap(self, stage):
        now = time.perf_counter()
        self.timings[stage] = self.timings.get(stage, 0.0) + (now - self.started) * 1000
        self.started = now


def blobs_around(grays):
    smoothed = [cv2.GaussianBlur(gray, (3, 3), 0).astype(np.int16) for gray in grays]
    return moving_blobs(*smoothed)


class ShotPipeline:
    def __init__(self, detector, court, right_row_offset, terminal_speed_m_s, correction, hit_frame, last_frame,
                 fit_every_frames=FIT_EVERY_FRAMES, predictor=None):
        self.detector = detector
        self.detector_takes_grays = getattr(detector, "input_kind", None) == GRAY_STACK_INPUT
        self.court = court
        self.right_row_offset = right_row_offset
        self.predictor = predictor or LandingPredictor(terminal_speed_m_s, correction, background=False)
        self.predictor.start_shot()
        self.fit_every_frames = fit_every_frames
        self.frames_since_fit = fit_every_frames
        self.fits_started = 0
        self.fits_waiting_for_worker = 0
        self.fit_ms = []
        self.hit_frame = hit_frame
        self.frames_to_track = range(hit_frame - 2, min(last_frame, hit_frame + FRAMES_AFTER_HIT))
        self.recent_grays = {eye: collections.deque(maxlen=FRAMES_NEEDED_PER_VIEW) for eye in EYES}
        self.frame_times = {}
        self.followers = {eye: CropFollower() for eye in EYES}
        self.found = {eye: {} for eye in EYES}
        self.stereo = []
        self.landing_cm = None
        self.landing_from_frame = None
        self.final_landing_cm = None

    def first_frame_needed(self):
        return self.frames_to_track.start - 1

    def last_frame_needed(self):
        return self.frames_to_track.stop

    def detector_input(self, eye):
        if self.detector_takes_grays:
            return self.recent_grays[eye]
        return [current_position_motion_view(*self.recent_grays[eye])]

    def detect_in_crop(self, images, center):
        height, width = images[0].shape[:2]
        left, top = crop_origin(*center, width, height)
        crops = [image[top:top + CROP_SIZE, left:left + CROP_SIZE] for image in images]
        crop = np.ascontiguousarray(np.dstack(crops) if len(crops) > 1 else crops[0])
        return [(x + left, y + top, confidence) for x, y, confidence in self.detector.detect(crop)]

    def crop_center(self, eye, frame):
        follower = self.followers[eye]
        blobs = blobs_around(self.recent_grays[eye]) if follower.needs_blobs(frame) else []
        height, width = self.recent_grays[eye][-1].shape[:2]
        return follower.crop_center(frame, blobs, width, height)

    def push(self, frame_index, stereo_frame, time_s):
        clock = StageClock()
        self.frame_times[frame_index] = time_s
        for eye in EYES:
            self.recent_grays[eye].append(plain_gray(crop_eye(stereo_frame, eye)))
        clock.lap("gray")
        middle = frame_index - 1
        if middle not in self.frames_to_track or len(self.recent_grays["left"]) < FRAMES_NEEDED_PER_VIEW:
            return None
        views = {eye: self.detector_input(eye) for eye in EYES}
        clock.lap("motion_view")

        left_center = self.crop_center("left", middle)
        clock.lap("aim_left")
        left_detections = self.detect_in_crop(views["left"], left_center)
        self.followers["left"].remember(middle, left_detections)
        self.found["left"][middle] = left_detections
        clock.lap("yolo_left")

        right_center = right_eye_aim(left_detections, self.right_row_offset) or self.crop_center("right", middle)
        clock.lap("aim_right")
        right_detections = self.detect_in_crop(views["right"], right_center)
        self.followers["right"].remember(middle, right_detections)
        self.found["right"][middle] = right_detections
        clock.lap("yolo_right")

        chain = longest_rising_chain(self.found["left"], self.hit_frame)
        self.stereo = match_in_right_eye(chain, self.found["right"], self.right_row_offset)
        clock.lap("track")

        flight = flight_points(self.court, self.stereo, self.hit_frame, self.frame_times)
        self.frames_since_fit += 1
        if flight is not None and self.frames_since_fit >= self.fit_every_frames:
            if self.predictor.submit(middle, *flight):
                self.frames_since_fit = 0
                self.fits_started += 1
            else:
                self.fits_waiting_for_worker += 1
        self.take_newest_landing()
        clock.lap("predict")
        return {"frame": middle, "predicted": flight is not None, "stages_ms": clock.timings,
                "landing_cm": None if self.landing_cm is None else [float(v) for v in self.landing_cm],
                "landing_from_frame": self.landing_from_frame}

    def take_newest_landing(self):
        result = self.predictor.poll()
        if result is not None:
            self.landing_from_frame, self.landing_cm, fit_ms = result
            self.fit_ms.append(fit_ms)

    def finish(self):
        flight = flight_points(self.court, self.stereo, self.hit_frame, self.frame_times)
        if flight is not None:
            self.final_landing_cm = self.predictor.final_landing_cm(*flight)
        return self.final_landing_cm
