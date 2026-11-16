import json
import os

import cv2
import numpy as np

MOTION_THRESHOLD = 12
BLOB_AREA_PX = (2, 400)
RISE_PER_FRAME_PX = (8, 70)
SIDEWAYS_PER_FRAME_PX = 25
MAX_SKIPPED_FRAMES = 2
FRAMES_AFTER_HIT = 40
RIGHT_EYE_ROW_TOLERANCE_PX = 10
DISPARITY_RANGE_PX = (1, 150)
DISPARITY_JUMP_PX = 12
LATEST_START_AFTER_HIT = 15
EYES = ("left", "right")


def moving_blobs(older, middle, newer):
    moving = np.minimum(np.abs(newer - middle), np.abs(middle - older)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats((moving > MOTION_THRESHOLD).astype(np.uint8))
    return [(float(centers[k][0]), float(centers[k][1]), int(stats[k, cv2.CC_STAT_AREA]))
            for k in range(1, count) if BLOB_AREA_PX[0] <= stats[k, cv2.CC_STAT_AREA] <= BLOB_AREA_PX[1]]


def next_rising_step(found, frame, x, y):
    for gap in range(1, MAX_SKIPPED_FRAMES + 2):
        options = []
        for next_x, next_y, _ in found.get(frame + gap, []):
            rise = (y - next_y) / gap
            sideways = abs(next_x - x) / gap
            if RISE_PER_FRAME_PX[0] <= rise <= RISE_PER_FRAME_PX[1] and sideways <= SIDEWAYS_PER_FRAME_PX:
                options.append((sideways, frame + gap, next_x, next_y))
        if options:
            _, next_frame, next_x, next_y = min(options)
            return next_frame, next_x, next_y
    return None


def longest_rising_chain(found, hit_frame):
    best = []
    for start_frame in sorted(found):
        if start_frame > hit_frame + LATEST_START_AFTER_HIT:
            break
        for x, y, _ in found[start_frame]:
            chain = [(start_frame, x, y)]
            while True:
                step = next_rising_step(found, *chain[-1])
                if step is None:
                    break
                chain.append(step)
            rise = chain[0][2] - chain[-1][2]
            if rise > (best[0][2] - best[-1][2] if best else 0):
                best = chain
    return best


def right_eye_candidates(right_blobs, x, y, right_row_offset):
    return [(x - right_x, right_x, right_y) for right_x, right_y, _ in right_blobs
            if abs(right_y - (y + right_row_offset)) <= RIGHT_EYE_ROW_TOLERANCE_PX
            and DISPARITY_RANGE_PX[0] <= x - right_x <= DISPARITY_RANGE_PX[1]]


def closest_candidate(candidates, previous_disparity, expected_right_y):
    if previous_disparity is None:
        return min(candidates, key=lambda candidate: abs(candidate[2] - expected_right_y))
    return min(candidates, key=lambda candidate: abs(candidate[0] - previous_disparity))


def match_in_right_eye(left_chain, right_found, right_row_offset):
    matches = []
    previous_disparity = None
    for frame, x, y in left_chain:
        candidates = right_eye_candidates(right_found.get(frame, []), x, y, right_row_offset)
        if previous_disparity is not None:
            candidates = [c for c in candidates if abs(c[0] - previous_disparity) <= DISPARITY_JUMP_PX]
        if not candidates:
            continue
        disparity, right_x, right_y = closest_candidate(candidates, previous_disparity, y + right_row_offset)
        matches.append([frame, round(x, 2), round(y, 2), round(right_x, 2), round(right_y, 2)])
        previous_disparity = disparity
    return matches


def calibrated_right_row_offset(session_dir):
    path = os.path.join(session_dir, "net_calibration.json")
    if not os.path.exists(path):
        return 10.0
    with open(path) as f:
        return json.load(f)["intrinsics"]["right_eye_vertical_offset_px"]
