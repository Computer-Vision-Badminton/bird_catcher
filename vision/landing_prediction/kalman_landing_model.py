import json
import os
import time

import numpy as np

from vision.calibration.calibrate_net import NOMINAL_BASELINE_M
from vision.flight.flight_model import CM_PER_M, GRAVITY_M_S2, INTEGRATION_STEP_S, MAX_FLIGHT_S, apply_correction

try:
    from vision.flight import flight_core
except ImportError:
    flight_core = None

KALMAN_LANDING_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalman_landing_settings.json")
SETTINGS_ONLY_IN_PYTHON = ("first_frame_after_hit",)


def load_kalman_landing_settings(path=KALMAN_LANDING_SETTINGS_PATH):
    with open(path) as f:
        return json.load(f)


def stereo_camera_from_calibration(calibration):
    intrinsics = calibration["intrinsics"]
    return flight_core.StereoCamera(
        focal_px=intrinsics["focal_px"], principal_point_px=intrinsics["principal_point_px"],
        right_row_offset_px=intrinsics["right_eye_vertical_offset_px"], distortion_k1_k2=intrinsics["lens_distortion_k1_k2"],
        baseline_m=intrinsics.get("baseline_m", NOMINAL_BASELINE_M), camera_from_court=calibration["camera_from_court"]["matrix"],
        translation_m=calibration["camera_from_court"]["translation_m"])


def kalman_settings_from(filter_setting):
    settings = flight_core.KalmanSettings()
    for name, value in filter_setting.items():
        if name not in SETTINGS_ONLY_IN_PYTHON:
            setattr(settings, name, value)
    settings.step_s = INTEGRATION_STEP_S
    settings.max_flight_s = MAX_FLIGHT_S
    settings.gravity = GRAVITY_M_S2
    return settings


def pixel_measurements(stereo_matches, hit_frame, frame_times, first_frame_after_hit):
    return [(int(frame), frame_times[int(frame)], left_x, left_y, left_x - right_x)
            for frame, left_x, left_y, right_x, _ in stereo_matches if frame - hit_frame >= first_frame_after_hit]


class KalmanLandingPredictor:
    def __init__(self, calibration, settings=None):
        if flight_core is None:
            raise RuntimeError("the Kalman landing needs the C++ flight core: run bash vision/flight/core_source/build.sh")
        settings = settings or load_kalman_landing_settings()
        self.camera = stereo_camera_from_calibration(calibration)
        self.kalman_settings = kalman_settings_from(settings["filter_setting"])
        self.first_frame_after_hit = settings["filter_setting"]["first_frame_after_hit"]
        self.correction = np.asarray(settings["correction"], float)
        self.rebuilds = 0
        self.start_shot()

    def start_shot(self):
        self.kalman = flight_core.KalmanFlight(self.camera, self.kalman_settings)
        self.fed = []
        self.newest_landing_m = None

    def feed(self, measurements):
        for _, time_s, u, v, disparity in measurements:
            landing = self.kalman.update(time_s, u, v, disparity)
            if landing is not None:
                self.newest_landing_m = landing
        self.fed.extend(measurements)

    def update(self, stereo_matches, hit_frame, frame_times):
        started = time.perf_counter()
        measurements = pixel_measurements(stereo_matches, hit_frame, frame_times, self.first_frame_after_hit)
        if measurements[:len(self.fed)] != self.fed:
            self.start_shot()
            self.rebuilds += 1
        self.feed(measurements[len(self.fed):])
        return self.corrected_landing_cm(), (time.perf_counter() - started) * 1000

    def corrected_landing_cm(self):
        if self.newest_landing_m is None:
            return None
        return apply_correction(self.correction, (self.newest_landing_m[:2] * CM_PER_M)[None, :])[0]
