#!/usr/bin/env python3
import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from scipy.optimize import least_squares

from frames import FrameSource

PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrate_net.html")
EYE_WIDTH, EYE_HEIGHT = 1280, 720
NOMINAL_FOCAL_PX = 700.0
FOCAL_RANGE_PX = (350.0, 1400.0)
DISTORTION_RANGE = ((-0.6, 0.3), (-0.5, 0.5))
MIN_OBSERVATIONS_FOR_DISTORTION = 10
LENS_FIELD_RADIUS_SQ = 1.4
TILT_SIGMA_DEG = 0.05
PRINCIPAL_ROW_RANGE_PX = (EYE_HEIGHT / 2 - 160, EYE_HEIGHT / 2 + 160)
NOMINAL_BASELINE_M = 0.12
CLICK_SIGMA_PX = 2.0
TAPE_MEASURE_SIGMA_M = 0.05

DOUBLES_HALF_WIDTH = 3.05
SINGLES_HALF_WIDTH = 2.59
NET_HEIGHT_AT_POST = 1.55
SHORT_SERVICE_LINE = 1.98
DOUBLES_LONG_SERVICE_LINE = 5.94
BACK_BOUNDARY = 6.70

COURT_AXES = {
    "units": "centimetres",
    "origin": "floor, directly under the net, on the middle line",
    "x": "distance from the middle line toward the sideline, positive toward the post on the right of the image",
    "y": "distance from the net toward the back line, positive on the camera's (landing) side, negative on the hitter's side",
    "z": "height above the floor",
}
CM_PER_M = 100.0


def internal_from_court_cm(court_cm):
    x_from_middle, y_from_net, height = court_cm
    return [y_from_net / CM_PER_M, x_from_middle / CM_PER_M, height / CM_PER_M]


def court_cm_from_internal(internal):
    return [internal[1] * CM_PER_M, internal[0] * CM_PER_M, internal[2] * CM_PER_M]


def court_point(name, hint, x_from_middle_cm, y_from_net_cm, height_cm):
    return {"name": name, "hint": hint, "court_cm": [x_from_middle_cm, y_from_net_cm, height_cm], "left": None, "right": None}


def default_points():
    post_x = DOUBLES_HALF_WIDTH * CM_PER_M
    post_top = NET_HEIGHT_AT_POST * CM_PER_M
    back_line = -BACK_BOUNDARY * CM_PER_M
    return [
        court_point("left_post_top", "where the net's top band meets the left post", -post_x, 0.0, post_top),
        court_point("left_post_base", "where the left post meets the floor", -post_x, 0.0, 0.0),
        court_point("right_post_top", "where the net's top band meets the right post", post_x, 0.0, post_top),
        court_point("right_post_base", "where the right post meets the floor", post_x, 0.0, 0.0),
        court_point("hitter_back_center", "hitter's middle line meets the hitter's back line", 0.0, back_line, 0.0),
        court_point("hitter_back_left", "hitter's back line meets the left doubles sideline", -post_x, back_line, 0.0),
        court_point("hitter_back_right", "hitter's back line meets the right doubles sideline", post_x, back_line, 0.0),
        court_point("hitter_service_center", "hitter's middle line meets the hitter's short service line", 0.0, -SHORT_SERVICE_LINE * CM_PER_M, 0.0),
    ]


def court_line_segments():
    segments = []
    for side in (1, -1):
        for x in (SHORT_SERVICE_LINE, DOUBLES_LONG_SERVICE_LINE, BACK_BOUNDARY):
            segments.append(([side * x, -DOUBLES_HALF_WIDTH, 0], [side * x, DOUBLES_HALF_WIDTH, 0]))
        segments.append(([side * SHORT_SERVICE_LINE, 0, 0], [side * BACK_BOUNDARY, 0, 0]))
    for y in (-DOUBLES_HALF_WIDTH, -SINGLES_HALF_WIDTH, SINGLES_HALF_WIDTH, DOUBLES_HALF_WIDTH):
        segments.append(([-BACK_BOUNDARY, y, 0], [BACK_BOUNDARY, y, 0]))
    segments.append(([0, -DOUBLES_HALF_WIDTH, 0], [0, DOUBLES_HALF_WIDTH, 0]))
    for y in (-DOUBLES_HALF_WIDTH, DOUBLES_HALF_WIDTH):
        segments.append(([0, y, 0], [0, y, NET_HEIGHT_AT_POST]))
    segments.append(([0, -DOUBLES_HALF_WIDTH, NET_HEIGHT_AT_POST], [0, DOUBLES_HALF_WIDTH, NET_HEIGHT_AT_POST]))
    return segments


class StereoModel:
    def __init__(self, focal, right_vertical_offset, k1=0.0, k2=0.0, principal_row=EYE_HEIGHT / 2):
        self.focal = focal
        self.right_vertical_offset = right_vertical_offset
        self.k1 = k1
        self.k2 = k2
        self.principal_row = principal_row

    def normalized(self, rvec, tvec, court_points, eye):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, float))
        camera_points = (rotation @ np.asarray(court_points, float).T).T + np.asarray(tvec, float)
        if eye == "right":
            camera_points = camera_points - np.array([NOMINAL_BASELINE_M, 0.0, 0.0])
        depth = camera_points[:, 2]
        return camera_points[:, 0] / depth, camera_points[:, 1] / depth, depth

    def lens_radius_sq(self, rvec, tvec, court_points, eye):
        x, y, _ = self.normalized(rvec, tvec, court_points, eye)
        return x * x + y * y

    def project(self, rvec, tvec, court_points, eye):
        x, y, depth = self.normalized(rvec, tvec, court_points, eye)
        radius_sq = x * x + y * y
        bend = 1 + self.k1 * radius_sq + self.k2 * radius_sq * radius_sq
        u = self.focal * x * bend + EYE_WIDTH / 2
        v = self.focal * y * bend + self.principal_row
        if eye == "right":
            v = v + self.right_vertical_offset
        return np.stack([u, v], axis=1), depth


COURT_TO_INTERNAL_AXES = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def camera_center_in_court(rvec, tvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return -rotation.T @ np.asarray(tvec, float)


def clicked(points, eye):
    chosen = [p for p in points if p.get(eye)]
    return (np.array([internal_from_court_cm(p["court_cm"]) for p in chosen], float),
            np.array([p[eye] for p in chosen], float), chosen)


def initial_poses(court, pixels):
    camera_matrix = np.array([[NOMINAL_FOCAL_PX, 0, EYE_WIDTH / 2], [0, NOMINAL_FOCAL_PX, EYE_HEIGHT / 2], [0, 0, 1]])
    flags = cv2.SOLVEPNP_IPPE if len(court) >= 4 and np.ptp(court[:, 0]) < 1e-6 else cv2.SOLVEPNP_SQPNP
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(court, pixels, camera_matrix, None, flags=flags)
    return [(r.ravel(), t.ravel()) for r, t in zip(rvecs, tvecs)]


def fit_pose(residual_builder, starts, lower, upper):
    best = None
    for start in starts:
        fit = least_squares(residual_builder, np.array(start, float), bounds=(lower, upper), method="trf")
        if best is None or fit.cost < best.cost:
            best = fit
    return best


def pitch_up_deg(rvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, float))
    forward = rotation.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))


def solve(points, measured_camera_cm, measured_tilt_deg=None):
    left_court, left_pixels, left_points = clicked(points, "left")
    right_court, right_pixels, right_points = clicked(points, "right")
    if len(left_court) < 3:
        return {"error": "click at least 3 points in the left eye (4+ is much better)"}
    has_measured_camera = measured_camera_cm is not None and all(v is not None for v in measured_camera_cm)
    measured_camera = internal_from_court_cm(measured_camera_cm) if has_measured_camera else None
    fit_focal = len(right_court) >= 2 or has_measured_camera
    fit_right_offset = len(right_court) >= 1
    fit_distortion = fit_focal and len(left_court) + len(right_court) >= MIN_OBSERVATIONS_FOR_DISTORTION
    has_measured_tilt = measured_tilt_deg is not None

    def unpack(params, with_distortion, with_principal_row=False):
        position = 6
        focal = NOMINAL_FOCAL_PX
        if fit_focal:
            focal, position = params[position], position + 1
        offset = 0.0
        if fit_right_offset:
            offset, position = params[position], position + 1
        k1, k2 = (params[position], params[position + 1]) if with_distortion else (0.0, 0.0)
        principal_row = params[-1] if with_principal_row else EYE_HEIGHT / 2
        return params[:3], params[3:6], StereoModel(focal, offset, k1, k2, principal_row)

    def residuals_for(with_distortion, with_principal_row=False):
        def residuals(params):
            rvec, tvec, model = unpack(params, with_distortion, with_principal_row)
            parts = [(model.project(rvec, tvec, left_court, "left")[0] - left_pixels).ravel() / CLICK_SIGMA_PX]
            if len(right_court):
                parts.append((model.project(rvec, tvec, right_court, "right")[0] - right_pixels).ravel() / CLICK_SIGMA_PX)
            if has_measured_camera:
                parts.append((camera_center_in_court(rvec, tvec) - np.array(measured_camera, float)) / TAPE_MEASURE_SIGMA_M)
            if with_principal_row:
                parts.append(np.array([(pitch_up_deg(rvec) - measured_tilt_deg) / TILT_SIGMA_DEG]))
            return np.concatenate(parts)
        return residuals

    def bounds(with_distortion):
        lower = [-np.inf] * 6 + ([FOCAL_RANGE_PX[0]] if fit_focal else []) + ([-50.0] if fit_right_offset else [])
        upper = [np.inf] * 6 + ([FOCAL_RANGE_PX[1]] if fit_focal else []) + ([50.0] if fit_right_offset else [])
        if with_distortion:
            lower += [DISTORTION_RANGE[0][0], DISTORTION_RANGE[1][0]]
            upper += [DISTORTION_RANGE[0][1], DISTORTION_RANGE[1][1]]
        return lower, upper

    best = None
    for rvec, tvec in initial_poses(left_court, left_pixels):
        start = list(rvec) + list(tvec) + ([NOMINAL_FOCAL_PX] if fit_focal else []) + ([0.0] if fit_right_offset else [])
        fit = fit_pose(residuals_for(False), [start], *bounds(False))
        rvec, tvec, model = unpack(fit.x, False)
        in_front = np.all(model.project(rvec, tvec, left_court, "left")[1] > 0)
        on_camera_side = camera_center_in_court(rvec, tvec)[0] > 0
        score = (not (in_front and on_camera_side), fit.cost)
        if best is None or score < best[0]:
            best = (score, fit)
    params = best[1].x
    if fit_distortion:
        starts = []
        for focal_start in (params[6], 500.0, 700.0, 900.0):
            start = list(params)
            start[6] = focal_start
            starts.append(start + [0.0, 0.0])
        params = fit_pose(residuals_for(True), starts, *bounds(True)).x
    use_principal_row = has_measured_tilt and fit_distortion
    if use_principal_row:
        lower, upper = bounds(True)
        params = fit_pose(residuals_for(True, True), [list(params) + [EYE_HEIGHT / 2]],
                          lower + [PRINCIPAL_ROW_RANGE_PX[0]], upper + [PRINCIPAL_ROW_RANGE_PX[1]]).x
    rvec, tvec, model = unpack(params, fit_distortion, use_principal_row)
    result = describe_solution(rvec, tvec, model, left_points, right_points, fit_focal, has_measured_camera)
    result["used_measured_tilt_deg"] = measured_tilt_deg if use_principal_row else None
    if has_measured_tilt and not use_principal_row:
        result["warnings"].append("Tilt was ignored: it needs the tripod measurement and at least 10 clicks (both eyes count).")
    return result


def camera_angles_deg(rotation):
    forward = rotation.T @ np.array([0.0, 0.0, 1.0])
    image_down = rotation.T @ np.array([0.0, 1.0, 0.0])
    yaw = np.degrees(np.arctan2(forward[1], -forward[0]))
    pitch = np.degrees(np.arcsin(np.clip(forward[2], -1, 1)))
    level_right = np.cross(forward, [0.0, 0.0, 1.0])
    level_down = np.cross(forward, level_right)
    roll = np.degrees(np.arctan2(np.dot(image_down, level_right), np.dot(image_down, level_down)))
    return {"yaw_from_facing_net": round(float(yaw), 2), "pitch_up": round(float(pitch), 2), "roll": round(float(roll), 2)}


def reprojection_report(rvec, tvec, model, chosen, eye):
    if not chosen:
        return []
    projected = model.project(rvec, tvec, [internal_from_court_cm(p["court_cm"]) for p in chosen], eye)[0]
    return [{"name": p["name"], "eye": eye, "clicked": p[eye],
             "projected": [round(float(pr[0]), 1), round(float(pr[1]), 1)],
             "error_px": round(float(np.hypot(*(np.array(p[eye]) - pr))), 2)} for p, pr in zip(chosen, projected)]


def overlay_polylines(rvec, tvec, model, eye):
    polylines = []
    for start, end in court_line_segments():
        samples = np.linspace(start, end, 60)
        pixels, depth = model.project(rvec, tvec, samples, eye)
        radius_sq = model.lens_radius_sq(rvec, tvec, samples, eye)
        run = []
        for (u, v), z, r2 in zip(pixels, depth, radius_sq):
            if z > 0.05 and r2 < LENS_FIELD_RADIUS_SQ and -2000 < u < 3300 and -2000 < v < 2700:
                run.append([round(float(u), 1), round(float(v), 1)])
            elif len(run) > 1:
                polylines.append(run)
                run = []
            else:
                run = []
        if len(run) > 1:
            polylines.append(run)
    return polylines


def all_in_net_plane(chosen):
    return all(abs(p["court_cm"][1]) < 1e-6 for p in chosen)


def solution_warnings(model, errors, fit_focal, has_measured_camera, left_points):
    warnings = []
    if not has_measured_camera and all_in_net_plane(left_points):
        warnings.append("All points are on the net and there is no tripod measurement: distance and focal length trade off, so the distance can be badly wrong. Add floor points on the hitter's court or enter the tape-measured position.")
    if fit_focal and min(abs(model.focal - bound) for bound in FOCAL_RANGE_PX) < 1:
        warnings.append(f"Focal length hit its limit ({model.focal:.0f} px): this solution is not trustworthy.")
    if len(left_points) < 4:
        warnings.append("Fewer than 4 points in the left eye: add more for a stable result.")
    if any(entry["error_px"] > 5 for entry in errors):
        warnings.append("Some points are off by more than 5 px: check those clicks or their court coordinates.")
    return warnings


def describe_solution(rvec, tvec, model, left_points, right_points, fit_focal, has_measured_camera):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, float))
    errors = reprojection_report(rvec, tvec, model, left_points, "left") + reprojection_report(rvec, tvec, model, right_points, "right")
    return {
        "warnings": solution_warnings(model, errors, fit_focal, has_measured_camera, left_points),
        "court_axes": COURT_AXES,
        "camera_position_court_cm": [round(float(v), 1) for v in court_cm_from_internal(camera_center_in_court(rvec, tvec))],
        "camera_angles_deg": camera_angles_deg(rotation),
        "camera_from_court": {
            "how_to_use": "camera_point_m = matrix @ [x, y, z] (court point in metres, x from middle line, y from net) + translation_m",
            "matrix": (rotation @ COURT_TO_INTERNAL_AXES).round(6).tolist(),
            "translation_m": [round(float(v), 4) for v in tvec],
        },
        "intrinsics": {
            "focal_px": round(float(model.focal), 1),
            "focal_was_fitted": fit_focal,
            "principal_point_px": [EYE_WIDTH / 2, round(float(model.principal_row), 1)],
            "baseline_m": NOMINAL_BASELINE_M,
            "right_eye_vertical_offset_px": round(float(model.right_vertical_offset), 2),
            "lens_distortion_k1_k2": [round(float(model.k1), 4), round(float(model.k2), 4)],
            "lens_distortion_model": "radial: pixel = focal * normalized * (1 + k1 r^2 + k2 r^4) + centre",
        },
        "used_tape_measured_camera": has_measured_camera,
        "reprojection": errors,
        "rms_error_px": round(float(np.sqrt(np.mean([e["error_px"] ** 2 for e in errors]))), 2) if errors else None,
        "overlay": {"left": overlay_polylines(rvec, tvec, model, "left"),
                    "right": overlay_polylines(rvec, tvec, model, "right")},
    }


class CalibrationStore:
    def __init__(self, session_dir):
        self.points_path = os.path.join(session_dir, "net_calibration_points.json")
        self.result_path = os.path.join(session_dir, "net_calibration.json")

    def load(self):
        if os.path.exists(self.points_path):
            with open(self.points_path) as f:
                return json.load(f)
        return {"points": default_points(), "measured_camera_cm": [None, None, None], "frame": 0}

    def save(self, state):
        with open(self.points_path, "w") as f:
            json.dump(state, f, indent=2)

    def save_result(self, state, source_name, result):
        stored = {key: value for key, value in result.items() if key != "overlay"}
        stored["source"] = source_name
        stored["frame"] = state.get("frame")
        stored["points"] = state["points"]
        stored["measured_camera_court_cm"] = state.get("measured_camera_cm")
        stored["measured_tilt_deg"] = state.get("measured_tilt_deg")
        with open(self.result_path, "w") as f:
            json.dump(stored, f, indent=2)


class ImageSource:
    def __init__(self, path):
        self.path = path
        self.is_video = not path.lower().endswith((".png", ".jpg", ".jpeg"))
        self.video = FrameSource(path) if self.is_video else None
        self.still = None if self.is_video else cv2.imread(path)
        if self.still is None and not self.is_video:
            sys.exit(f"cannot read {path}")

    def frame_count(self):
        return self.video.frame_count if self.is_video else 1

    def jpeg(self, frame_index):
        if self.is_video:
            with self.video.lock:
                image = self.video.read(frame_index)
        else:
            image = self.still
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded.tobytes()


def make_handler(source, store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value):
            self.send_body(json.dumps(value).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                with open(PAGE_PATH, "rb") as f:
                    self.send_body(f.read(), "text/html; charset=utf-8")
            elif url.path == "/state":
                self.send_json({"source": os.path.basename(source.path), "is_video": source.is_video,
                                "frames": source.frame_count(), "state": store.load()})
            elif url.path == "/image":
                self.send_body(source.jpeg(int(query.get("frame", ["0"])[0])), "image/jpeg")
            else:
                self.send_body(b"not found", "text/plain", 404)

        def do_POST(self):
            state = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            store.save(state)
            if self.path == "/save":
                self.send_json({"saved": True})
            elif self.path == "/solve":
                result = solve(state["points"], state.get("measured_camera_cm"), state.get("measured_tilt_deg"))
                if "error" not in result:
                    store.save_result(state, os.path.basename(source.path), result)
                    result["saved_to"] = store.result_path
                self.send_json(result)
            else:
                self.send_body(b"not found", "text/plain", 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Click the net in a stereo frame to find where the camera sits on the court.")
    parser.add_argument("source", help="calibration.png or a take .avi (pick the frame in the page)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source)
    source = ImageSource(source_path)
    store = CalibrationStore(os.path.dirname(source_path))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(source, store))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"net calibration at {url}  (Ctrl+C to stop)")
    print(f"result saved to {store.result_path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
