import json
import os

import cv2
import numpy as np

from vision.calibration.calibrate_net import COURT_TO_INTERNAL_AXES, EYE_WIDTH, NOMINAL_BASELINE_M, court_cm_from_internal


class StereoCourt:
    def __init__(self, calibration):
        self.calibration = calibration
        intrinsics = calibration["intrinsics"]
        focal = intrinsics["focal_px"]
        center_column, center_row = intrinsics["principal_point_px"]
        right_row = center_row + intrinsics["right_eye_vertical_offset_px"]
        self.left_matrix = np.array([[focal, 0, center_column], [0, focal, center_row], [0, 0, 1]])
        self.right_matrix = np.array([[focal, 0, center_column], [0, focal, right_row], [0, 0, 1]])
        self.distortion = np.array(list(intrinsics["lens_distortion_k1_k2"]) + [0.0, 0.0])
        self.baseline_m = intrinsics.get("baseline_m", NOMINAL_BASELINE_M)
        matrix = np.array(calibration["camera_from_court"]["matrix"])
        self.rotation = matrix @ COURT_TO_INTERNAL_AXES
        self.translation = np.array(calibration["camera_from_court"]["translation_m"])

    @classmethod
    def from_session(cls, session_dir):
        with open(os.path.join(session_dir, "net_calibration.json")) as f:
            return cls(json.load(f))

    def normalized(self, pixels, camera_matrix):
        points = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, camera_matrix, self.distortion).reshape(-1, 2)

    def camera_points_m(self, left_pixels, right_pixels):
        left = self.normalized(left_pixels, self.left_matrix)
        right = self.normalized(right_pixels, self.right_matrix)
        disparity = left[:, 0] - right[:, 0]
        depth = self.baseline_m / disparity
        height_in_image = (left[:, 1] + right[:, 1]) / 2
        return np.column_stack([left[:, 0] * depth, height_in_image * depth, depth])

    def court_points_cm(self, left_pixels, right_pixels):
        camera_points = self.camera_points_m(left_pixels, right_pixels)
        internal = (self.rotation.T @ (camera_points - self.translation).T).T
        return np.array([court_cm_from_internal(point) for point in internal])
