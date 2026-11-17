import csv
import os
import sys
import threading

import cv2

SEQUENTIAL_READ_LIMIT = 30


class FrameSource:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit(f"cannot open {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.container_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.next_index = 0
        self.recent = {}
        self.lock = threading.Lock()

    def read(self, index):
        index = max(0, min(self.frame_count - 1, index))
        if index in self.recent:
            return self.recent[index]
        gap = index - self.next_index
        if 0 <= gap <= SEQUENTIAL_READ_LIMIT:
            for _ in range(gap):
                self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {index}")
        self.next_index = index + 1
        self.recent = {kept: image for kept, image in self.recent.items() if index - 2 <= kept < index}
        self.recent[index] = frame
        return frame


def crop_eye(frame, view):
    half = frame.shape[1] // 2
    if view == "left":
        return frame[:, :half]
    if view == "right":
        return frame[:, half:]
    return frame


def plain_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_frame_times(video_path):
    csv_path = os.path.splitext(video_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        return [], []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [float(row["t_mono_s"]) for row in rows], [row["t_wall_iso"] for row in rows]
