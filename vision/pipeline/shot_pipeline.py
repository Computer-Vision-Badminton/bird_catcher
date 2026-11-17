import collections
import time

import cv2
import numpy as np

from vision.tracking.bird_tracker import EYES, FRAMES_AFTER_HIT, longest_rising_chain, match_in_right_eye, moving_blobs_in_uint8
from vision.detection.crop_tracking import CROP_SIZE, FOLLOW_CROP_SIZE, CropFollower, crop_origin, right_crop_center
from vision.detection.detectors import GRAY_STACK_INPUT
from vision.flight.flight_model import flight_points
from vision.camera.frames import crop_eye, plain_gray
from vision.landing_prediction.kalman_landing_model import KalmanLandingPredictor
from vision.landing_prediction.landing_worker import LandingPredictor
from vision.detection.motion_view import current_position_motion_view

STAGES = ("gray", "motion_view", "aim", "yolo", "track", "predict")
FRAMES_NEEDED_PER_VIEW = 3
FIT_EVERY_FRAMES = 1
SEARCH_WITH_FOLLOW_ALWAYS = "always"
SEARCH_WITH_FOLLOW_ONCE_SEEN = "once-seen"
SEARCH_WITH_FOLLOW_CHOICES = (SEARCH_WITH_FOLLOW_ALWAYS, SEARCH_WITH_FOLLOW_ONCE_SEEN)
LANDING_BY_FIT = "fit"
LANDING_BY_KALMAN = "kalman"
LANDING_METHODS = (LANDING_BY_FIT, LANDING_BY_KALMAN)
FRAMES_KEPT_BLURRED = 4


class StageClock:
    def __init__(self):
        self.timings = {}
        self.started = time.perf_counter()

    def lap(self, stage):
        now = time.perf_counter()
        self.timings[stage] = self.timings.get(stage, 0.0) + (now - self.started) * 1000
        self.started = now


def blurred_for_blobs(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)


def blobs_around(grays):
    return moving_blobs_in_uint8(*[blurred_for_blobs(gray) for gray in grays])


class BlurredFrameCache:
    def __init__(self):
        self.grays_and_blurred = collections.deque(maxlen=FRAMES_KEPT_BLURRED)

    def blurred(self, gray):
        for kept_gray, kept_blurred in self.grays_and_blurred:
            if kept_gray is gray:
                return kept_blurred
        blurred = blurred_for_blobs(gray)
        self.grays_and_blurred.append((gray, blurred))
        return blurred

    def blobs_around(self, grays):
        return moving_blobs_in_uint8(*[self.blurred(gray) for gray in grays])


def blobs_near(grays, center):
    height, width = grays[-1].shape[:2]
    left, top = crop_origin(*center, width, height)
    window = [gray[top:top + CROP_SIZE, left:left + CROP_SIZE] for gray in grays]
    return [(x + left, y + top, area) for x, y, area in blobs_around(window)]


def detections_in_frame(detections_per_crop, origins):
    return [[(x + left, y + top, confidence) for x, y, confidence in detections]
            for detections, (left, top) in zip(detections_per_crop, origins)]


class ShotPipeline:
    def __init__(self, detector, court, right_row_offset, terminal_speed_m_s, correction, hit_frame, last_frame,
                 fit_every_frames=FIT_EVERY_FRAMES, predictor=None, follow_detector=None, search_with_follow_detector=None,
                 overlap_detection=False, landing_method=LANDING_BY_FIT):
        self.detector = detector
        self.follow_detector = follow_detector
        self.search_with_follow_detector = search_with_follow_detector
        self.detector_takes_grays = getattr(detector, "input_kind", None) == GRAY_STACK_INPUT
        self.court = court
        self.right_row_offset = right_row_offset
        self.landing_method = landing_method
        if landing_method == LANDING_BY_KALMAN:
            self.predictor = KalmanLandingPredictor(court.calibration)
        else:
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
        self.left_follower = CropFollower()
        self.blurred_left_frames = BlurredFrameCache()
        self.found = {eye: {} for eye in EYES}
        self.stereo = []
        self.landing_cm = None
        self.landing_from_frame = None
        self.final_landing_cm = None
        self.overlap_detection = overlap_detection
        self.pending_detection = None
        self.newest_frame_arrived_at = None

    def first_frame_needed(self):
        return self.frames_to_track.start - 1

    def last_frame_needed(self):
        return self.frames_to_track.stop

    def detector_input(self, eye):
        if self.detector_takes_grays:
            return self.recent_grays[eye]
        return [current_position_motion_view(*self.recent_grays[eye])]

    def detector_and_crop_size(self, frame):
        if self.follow_detector is not None and self.follow_detector_fits(frame):
            return self.follow_detector, FOLLOW_CROP_SIZE
        return self.detector, CROP_SIZE

    def follow_detector_fits(self, frame):
        if not self.left_follower.needs_blobs(frame):
            return True
        if self.search_with_follow_detector == SEARCH_WITH_FOLLOW_ALWAYS:
            return True
        return self.search_with_follow_detector == SEARCH_WITH_FOLLOW_ONCE_SEEN and bool(self.left_follower.recent)

    def cut_crops(self, views_and_centers, crop_size):
        crops, origins = [], []
        for images, center in views_and_centers:
            height, width = images[0].shape[:2]
            left, top = crop_origin(*center, width, height, crop_size)
            window = [image[top:top + crop_size, left:left + crop_size] for image in images]
            crops.append(np.ascontiguousarray(np.dstack(window) if len(window) > 1 else window[0]))
            origins.append((left, top))
        return crops, origins

    def detect_in_crops(self, views_and_centers, detector, crop_size):
        crops, origins = self.cut_crops(views_and_centers, crop_size)
        return detections_in_frame(detector.detect_batch(crops), origins)

    def left_crop_center(self, frame):
        grays = self.recent_grays["left"]
        height, width = grays[-1].shape[:2]
        if not self.left_follower.needs_blobs(frame):
            return self.left_follower.crop_center(frame, [], width, height)
        heading = self.left_follower.last_seen_heading(frame)
        blobs = self.blurred_left_frames.blobs_around(grays) if heading is None else blobs_near(grays, heading)
        return self.left_follower.crop_center(frame, blobs, width, height, without_blobs=heading)

    def take_frame(self, frame_index, stereo_frame, time_s):
        self.newest_frame_arrived_at = time.perf_counter()
        self.frame_times[frame_index] = time_s
        for eye in EYES:
            self.recent_grays[eye].append(plain_gray(crop_eye(stereo_frame, eye)))

    def ready_to_detect(self, frame):
        return frame in self.frames_to_track and len(self.recent_grays["left"]) >= FRAMES_NEEDED_PER_VIEW

    def aimed_views(self, frame, clock):
        views = {eye: self.detector_input(eye) for eye in EYES}
        clock.lap("motion_view")
        left_center = self.left_crop_center(frame)
        right_center = right_crop_center(left_center, self.right_row_offset)
        detector, crop_size = self.detector_and_crop_size(frame)
        clock.lap("aim")
        return [(views["left"], left_center), (views["right"], right_center)], detector, crop_size

    def remember_detections(self, frame, left_detections, right_detections):
        self.left_follower.remember(frame, left_detections)
        self.found["left"][frame] = left_detections
        self.found["right"][frame] = right_detections

    def push(self, frame_index, stereo_frame, time_s):
        if self.overlap_detection:
            return self.push_overlapped(frame_index, stereo_frame, time_s)
        clock = StageClock()
        self.take_frame(frame_index, stereo_frame, time_s)
        clock.lap("gray")
        middle = frame_index - 1
        if not self.ready_to_detect(middle):
            return None
        inputs_arrived_at = self.newest_frame_arrived_at
        views_and_centers, detector, crop_size = self.aimed_views(middle, clock)
        self.remember_detections(middle, *self.detect_in_crops(views_and_centers, detector, crop_size))
        clock.lap("yolo")
        return self.track_and_predict(middle, crop_size, inputs_arrived_at, None, clock)

    def push_overlapped(self, frame_index, stereo_frame, time_s):
        clock = StageClock()
        self.take_frame(frame_index, stereo_frame, time_s)
        clock.lap("gray")
        detected = self.finish_pending_detection(clock)
        middle = frame_index - 1
        if self.ready_to_detect(middle):
            self.start_detection(middle, clock)
        if detected is None:
            return None
        return self.track_and_predict(*detected, clock)

    def start_detection(self, frame, clock):
        inputs_arrived_at = self.newest_frame_arrived_at
        views_and_centers, detector, crop_size = self.aimed_views(frame, clock)
        crops, origins = self.cut_crops(views_and_centers, crop_size)
        self.pending_detection = (frame, crop_size, origins, inputs_arrived_at, detector.start_batch(crops))
        clock.lap("yolo")

    def finish_pending_detection(self, clock):
        if self.pending_detection is None:
            return None
        frame, crop_size, origins, inputs_arrived_at, job = self.pending_detection
        self.pending_detection = None
        self.remember_detections(frame, *detections_in_frame(job.wait(), origins))
        clock.lap("yolo")
        return frame, crop_size, inputs_arrived_at, job.call_ms

    def take_known_frame_time(self, frame_index, time_s):
        self.frame_times[frame_index] = time_s

    def take_known_grays(self, grays_by_eye):
        for eye in EYES:
            self.recent_grays[eye].extend(grays_by_eye[eye])
        self.newest_frame_arrived_at = time.perf_counter()

    def take_running_detection(self, frame, crop_size, origins, job):
        self.pending_detection = (frame, crop_size, origins, time.perf_counter(), job)

    def push_detected(self, frame, left_detections, right_detections, crop_size):
        if frame not in self.frames_to_track:
            return None
        clock = StageClock()
        self.remember_detections(frame, left_detections, right_detections)
        clock.lap("yolo")
        return self.track_and_predict(frame, crop_size, clock.started, None, clock)

    def drain(self):
        clock = StageClock()
        detected = self.finish_pending_detection(clock)
        return None if detected is None else self.track_and_predict(*detected, clock)

    def track_and_predict(self, frame, crop_size, inputs_arrived_at, detector_call_ms, clock):
        chain = longest_rising_chain(self.found["left"], self.hit_frame)
        self.stereo = match_in_right_eye(chain, self.found["right"], self.right_row_offset)
        clock.lap("track")
        predicted = self.update_kalman_landing(frame) if self.landing_method == LANDING_BY_KALMAN else self.update_fitted_landing(frame)
        clock.lap("predict")
        return {"frame": frame, "predicted": predicted, "stages_ms": clock.timings, "crop_size": crop_size,
                "landing_cm": None if self.landing_cm is None else [float(v) for v in self.landing_cm],
                "landing_from_frame": self.landing_from_frame,
                "ready_after_inputs_ms": (time.perf_counter() - inputs_arrived_at) * 1000,
                "detector_call_ms": detector_call_ms}

    def update_fitted_landing(self, frame):
        flight = flight_points(self.court, self.stereo, self.hit_frame, self.frame_times)
        self.frames_since_fit += 1
        if flight is not None and self.frames_since_fit >= self.fit_every_frames:
            if self.predictor.submit(frame, *flight):
                self.frames_since_fit = 0
                self.fits_started += 1
            else:
                self.fits_waiting_for_worker += 1
        self.take_newest_landing()
        return flight is not None

    def update_kalman_landing(self, frame):
        landing_cm, update_ms = self.predictor.update(self.stereo, self.hit_frame, self.frame_times)
        self.fits_started += 1
        self.fit_ms.append(update_ms)
        if landing_cm is not None:
            self.landing_from_frame, self.landing_cm = frame, landing_cm
        return landing_cm is not None

    def take_newest_landing(self):
        result = self.predictor.poll()
        if result is not None:
            self.landing_from_frame, self.landing_cm, fit_ms = result
            self.fit_ms.append(fit_ms)

    def finish(self):
        self.drain()
        if self.landing_method == LANDING_BY_KALMAN:
            self.final_landing_cm = self.predictor.update(self.stereo, self.hit_frame, self.frame_times)[0]
            return self.final_landing_cm
        flight = flight_points(self.court, self.stereo, self.hit_frame, self.frame_times)
        if flight is not None:
            self.final_landing_cm = self.predictor.final_landing_cm(*flight)
        return self.final_landing_cm
