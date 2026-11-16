import cv2
import numpy as np

MOTION_BOOST = 4
CONTEXT_BRIGHTNESS = 0.45
HIGHLIGHT_GREEN_SHARE = 0.75


def current_position_motion_view(previous, current, upcoming):
    smooth = [cv2.GaussianBlur(image, (3, 3), 0).astype(np.int16) for image in (previous, current, upcoming)]
    moved_now = np.minimum(np.abs(smooth[1] - smooth[0]), np.abs(smooth[1] - smooth[2])).astype(np.uint8)
    highlight = cv2.convertScaleAbs(moved_now, alpha=MOTION_BOOST)
    context = cv2.convertScaleAbs(current, alpha=CONTEXT_BRIGHTNESS)
    green = cv2.max(context, cv2.convertScaleAbs(highlight, alpha=HIGHLIGHT_GREEN_SHARE))
    red = cv2.max(context, highlight)
    return cv2.merge([context, green, red])
