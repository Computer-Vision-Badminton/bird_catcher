import numpy as np

from vision.tracking.bird_tracker import DISPARITY_RANGE_PX

CROP_SIZE = 640
FOLLOW_CROP_SIZE = 320
FRAMES_TO_KEEP_FOLLOWING = 2
MIN_CONFIDENCE = 0.25


def crop_origin(center_x, center_y, width, height, size=CROP_SIZE):
    left = int(np.clip(round(center_x - size / 2), 0, width - size))
    top = int(np.clip(round(center_y - size / 2), 0, height - size))
    return left, top


def extrapolated_position(detections, frame):
    if len(detections) >= 2:
        (f0, x0, y0), (f1, x1, y1) = detections[-2], detections[-1]
        steps = (frame - f1) / (f1 - f0)
        return x1 + (x1 - x0) * steps, y1 + (y1 - y0) * steps
    return detections[-1][1], detections[-1][2]


class CropFollower:
    def __init__(self):
        self.recent = []

    def recent_detections(self, frame):
        return [(f, x, y) for f, x, y in self.recent if frame - f <= FRAMES_TO_KEEP_FOLLOWING + 1]

    def needs_blobs(self, frame):
        return not self.recent_detections(frame)

    def crop_center(self, frame, blobs, width, height, without_blobs=None):
        recent = self.recent_detections(frame)
        if recent:
            return extrapolated_position(recent, frame)
        if blobs:
            return min(blobs, key=lambda blob: blob[1])[:2]
        return without_blobs or (width / 2, height / 2)

    def last_seen_heading(self, frame):
        return extrapolated_position(self.recent, frame) if self.recent else None

    def remember(self, frame, detections):
        if not detections:
            return
        expected = self.recent[-1][1:] if self.recent else None
        x, y, _ = (min(detections, key=lambda d: (d[0] - expected[0]) ** 2 + (d[1] - expected[1]) ** 2) if expected
                   else max(detections, key=lambda d: d[2]))
        self.recent = (self.recent + [(frame, x, y)])[-2:]


def right_eye_aim(left_detections, right_row_offset):
    if not left_detections:
        return None
    x, y, _ = max(left_detections, key=lambda detection: detection[2])
    return x - sum(DISPARITY_RANGE_PX) / 2, y + right_row_offset


def right_crop_center(left_center, right_row_offset):
    x, y = left_center
    return x - sum(DISPARITY_RANGE_PX) / 2, y + right_row_offset
