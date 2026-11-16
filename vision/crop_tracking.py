import numpy as np

from bird_tracker import DISPARITY_RANGE_PX

CROP_SIZE = 640
FRAMES_TO_KEEP_FOLLOWING = 2
MIN_CONFIDENCE = 0.25


def crop_origin(center_x, center_y, width, height):
    left = int(np.clip(round(center_x - CROP_SIZE / 2), 0, width - CROP_SIZE))
    top = int(np.clip(round(center_y - CROP_SIZE / 2), 0, height - CROP_SIZE))
    return left, top


class CropFollower:
    def __init__(self):
        self.recent = []

    def recent_detections(self, frame):
        return [(f, x, y) for f, x, y in self.recent if frame - f <= FRAMES_TO_KEEP_FOLLOWING + 1]

    def needs_blobs(self, frame):
        return not self.recent_detections(frame)

    def crop_center(self, frame, blobs, width, height):
        recent = self.recent_detections(frame)
        if len(recent) >= 2:
            (f0, x0, y0), (f1, x1, y1) = recent[-2], recent[-1]
            steps = (frame - f1) / (f1 - f0)
            return x1 + (x1 - x0) * steps, y1 + (y1 - y0) * steps
        if recent:
            return recent[-1][1], recent[-1][2]
        if blobs:
            return min(blobs, key=lambda blob: blob[1])[:2]
        return width / 2, height / 2

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
