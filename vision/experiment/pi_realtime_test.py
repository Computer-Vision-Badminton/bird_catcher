#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court_3d import StereoCourt
from detectors import DETECTOR_KINDS, INPUT_KINDS, MOTION_VIEW_INPUT, load_detector
from flight_model import FLIGHT_ENGINE
from landing_worker import LandingPredictor
from pipeline import FIT_EVERY_FRAMES, STAGES, ShotPipeline

CAMERA_FPS = 60
FRAME_BUDGET_MS = 1000 / CAMERA_FPS
WARMUP_CROPS = 5
CHECKPOINTS_AFTER_HIT = (10, 15, 20)


def warm_up(detector):
    blank = np.zeros((640, 640, 3), np.uint8)
    for _ in range(WARMUP_CROPS):
        detector.detect(blank)


def run_shot(package_dir, entry, detector, court, manifest, predictor, fit_every):
    pipeline = ShotPipeline(detector, court, manifest["right_row_offset_px"], manifest["terminal_speed_m_s"],
                            manifest["correction"], entry["hit_frame"], entry["end_frame"] + manifest["frames_after_hit"],
                            fit_every, predictor)
    clip = cv2.VideoCapture(os.path.join(package_dir, "clips", entry["clip"]))
    processed, decode_ms = [], []
    first_prediction_frame = None
    for offset, time_s in enumerate(entry["frame_times_s"]):
        started = time.perf_counter()
        ok, stereo_frame = clip.read()
        decode_ms.append((time.perf_counter() - started) * 1000)
        if not ok:
            raise SystemExit(f"{entry['clip']}: could not read frame {offset}")
        result = pipeline.push(entry["first_frame"] + offset, stereo_frame, time_s)
        if result is None:
            continue
        processed.append(result)
        if result["predicted"] and first_prediction_frame is None:
            first_prediction_frame = result["frame"]
    final = pipeline.finish()
    landing = None if final is None else [round(float(v), 1) for v in final]
    return processed, decode_ms, landing, first_prediction_frame, pipeline


def estimate_at(processed, frame):
    result = next((r for r in processed if r["frame"] == frame), None)
    return None if result is None else result["landing_cm"]


def apply_landing_tuning(manifest, tuning_path):
    with open(tuning_path) as f:
        tuning = json.load(f)
    manifest["terminal_speed_m_s"] = tuning["terminal_speed_m_s"]
    manifest["correction"] = tuning["correction"]
    offline = {entry["shot"]: entry["corrected_cm"] for entry in tuning["test_shots"]}
    for entry in manifest["shots"]:
        entry["mac_corrected_cm"] = offline.get(entry["shot"])


def distance_cm(a, b):
    return None if a is None or b is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def summary(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def print_report(stage_stats, total_stats, decode_stats, shot_rows, frames_over_budget, frame_count, predict_on_frames, fit_report):
    print(f"\nper processed stereo frame, {frame_count} frames over {len(shot_rows)} shots (budget {FRAME_BUDGET_MS:.1f} ms at {CAMERA_FPS} fps)")
    print(f"  {'stage':<14}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}  ms")
    for stage in STAGES:
        s = stage_stats[stage]
        print(f"  {stage:<14}{s['mean']:9.2f}{s['median']:9.2f}{s['p95']:9.2f}{s['max']:9.2f}")
    print(f"  {'TOTAL':<14}{total_stats['mean']:9.2f}{total_stats['median']:9.2f}{total_stats['p95']:9.2f}{total_stats['max']:9.2f}")
    print(f"  {'(video decode)':<14}{decode_stats['mean']:9.2f}{decode_stats['median']:9.2f}{decode_stats['p95']:9.2f}{decode_stats['max']:9.2f}  not in TOTAL")
    print(f"\n  mean {total_stats['mean']:.1f} ms -> {1000 / total_stats['mean']:.1f} fps; "
          f"{frames_over_budget}/{frame_count} frames over {FRAME_BUDGET_MS:.1f} ms")
    predict_mean = np.mean(predict_on_frames) if predict_on_frames else 0.0
    without_predict = total_stats["mean"] - stage_stats["predict"]["mean"]
    print(f"  predict stage on frames with enough points: {predict_mean:.1f} ms mean; everything except predict: {without_predict:.1f} ms mean per frame")
    print(f"\n  flight fit ({fit_report['engine']}): {fit_report['mode']}, every {fit_report['every']} frames; {fit_report['fits']} fits, "
          f"{fit_report['waited']} frames where a fit was due but the worker was still busy")
    if fit_report["fit_ms"]:
        print(f"  each fit took {fit_report['fit_ms']['mean']:.1f} ms mean, {fit_report['fit_ms']['p95']:.1f} ms p95"
              f"{' (on another core, not in TOTAL)' if fit_report['mode'] == 'background' else ''}")
    for after_hit in CHECKPOINTS_AFTER_HIT:
        misses = [row["checkpoint_miss_cm"][str(after_hit)] for row in shot_rows if row["checkpoint_miss_cm"][str(after_hit)] is not None]
        without = len(shot_rows) - len(misses)
        if misses:
            print(f"  estimate available at hit +{after_hit:>2} frames: mean miss {np.mean(misses):5.1f} cm, median {np.median(misses):5.1f} cm"
                  f"{f', {without} shots had no estimate yet' if without else ''}")
    print(f"\n  {'shot':>4}{'frames':>8}{'mean ms':>9}{'p95 ms':>8}{'1st fit':>9}{'landing x, y cm':>18}{'miss':>7}{'vs Mac':>8}")
    for row in shot_rows:
        landing = "none" if row["landing_cm"] is None else f"{row['landing_cm'][0]:.0f}, {row['landing_cm'][1]:.0f}"
        miss = "-" if row["miss_cm"] is None else f"{row['miss_cm']:.0f}"
        vs_mac = "-" if row["vs_mac_cm"] is None else f"{row['vs_mac_cm']:.1f}"
        first_fit = "-" if row["first_prediction_after_hit"] is None else f"+{row['first_prediction_after_hit']}"
        print(f"  {row['shot']:>4}{row['frames']:>8}{row['mean_ms']:>9.1f}{row['p95_ms']:>8.1f}{first_fit:>9}{landing:>18}{miss:>7}{vs_mac:>8}")
    misses = [row["miss_cm"] for row in shot_rows if row["miss_cm"] is not None]
    if misses:
        print(f"\n  landing miss: mean {np.mean(misses):.1f} cm, median {np.median(misses):.1f} cm "
              f"(the vs Mac column compares with the offline prediction of each shot)")


def main():
    parser = argparse.ArgumentParser(description="Run the whole bird pipeline frame by frame on the packaged test shots and time every stage.")
    parser.add_argument("--package", default="pi_test_package")
    parser.add_argument("--detector", choices=DETECTOR_KINDS, default="ncnn")
    parser.add_argument("--model", required=True, help="bird_yolo11n_v1_640_ncnn_model folder, bird_yolo11n_v1.pt, or the .hef file")
    parser.add_argument("--hailo-input-bgr", action="store_true", help="feed the Hailo BGR instead of RGB (match how the HEF was calibrated)")
    parser.add_argument("--shots", type=int, nargs="+", help="only these shot numbers")
    parser.add_argument("--input", choices=INPUT_KINDS, default=MOTION_VIEW_INPUT, help="gray-stack: the model builds the motion view itself from 3 gray crops")
    parser.add_argument("--fit-every", type=int, default=FIT_EVERY_FRAMES, help="start a flight fit every N frames")
    parser.add_argument("--fit-background", action="store_true", help="run the flight fit in a separate process on another core instead of inside the frame loop")
    parser.add_argument("--landing-tuning", help="landing_experiment_*.json in the package tuned on this model's tracks (terminal speed + correction)")
    args = parser.parse_args()

    with open(os.path.join(args.package, "manifest.json")) as f:
        manifest = json.load(f)
    if args.landing_tuning:
        apply_landing_tuning(manifest, os.path.join(args.package, args.landing_tuning))
    with open(os.path.join(args.package, "net_calibration.json")) as f:
        court = StereoCourt(json.load(f))
    predictor = LandingPredictor(manifest["terminal_speed_m_s"], manifest["correction"], background=args.fit_background)
    detector = load_detector(args.detector, args.model, args.hailo_input_bgr, args.input)
    warm_up(detector)

    all_frames, all_decode, shot_rows, all_fit_ms = [], [], [], []
    fits = waited = 0
    for entry in manifest["shots"]:
        if args.shots and entry["shot"] not in args.shots:
            continue
        processed, decode_ms, landing, first_prediction_frame, pipeline = run_shot(args.package, entry, detector, court, manifest, predictor, args.fit_every)
        all_fit_ms.extend(pipeline.fit_ms)
        fits += pipeline.fits_started
        waited += pipeline.fits_waiting_for_worker
        totals = [sum(frame["stages_ms"].values()) for frame in processed]
        all_frames.extend(processed)
        all_decode.extend(decode_ms)
        shot_rows.append({
            "shot": entry["shot"], "frames": len(processed), "stereo_points": len(pipeline.stereo),
            "mean_ms": float(np.mean(totals)), "p95_ms": float(np.percentile(totals, 95)),
            "first_prediction_after_hit": None if first_prediction_frame is None else first_prediction_frame - entry["hit_frame"],
            "landing_cm": landing, "miss_cm": distance_cm(landing, entry["actual_cm"]),
            "vs_mac_cm": distance_cm(landing, entry["mac_corrected_cm"]),
            "checkpoint_miss_cm": {str(k): distance_cm(estimate_at(processed, entry["hit_frame"] + k), entry["actual_cm"]) for k in CHECKPOINTS_AFTER_HIT},
        })
        print(f"shot {entry['shot']:>3}: {len(processed)} frames, {np.mean(totals):.1f} ms per frame, landing {landing}")

    stage_stats = {stage: summary([frame["stages_ms"].get(stage, 0.0) for frame in all_frames]) for stage in STAGES}
    totals = [sum(frame["stages_ms"].values()) for frame in all_frames]
    total_stats = summary(totals)
    decode_stats = summary(all_decode)
    frames_over_budget = int(sum(total > FRAME_BUDGET_MS for total in totals))
    predict_on_frames = [frame["stages_ms"]["predict"] for frame in all_frames if frame["predicted"]]
    fit_report = {"mode": "background" if args.fit_background else "inline", "engine": FLIGHT_ENGINE, "every": args.fit_every, "fits": fits, "waited": waited,
                  "fit_ms": summary(all_fit_ms) if all_fit_ms else None}
    print_report(stage_stats, total_stats, decode_stats, shot_rows, frames_over_budget, len(all_frames), predict_on_frames, fit_report)

    model_name = os.path.splitext(os.path.basename(os.path.normpath(args.model)))[0]
    fit_name = (f"background{args.fit_every}" if args.fit_background else f"inline{args.fit_every}") + ("_cpp" if FLIGHT_ENGINE == "c++" else "")
    run_name = f"{platform.node().split('.')[0]}_{args.detector}_{model_name}_{fit_name}".replace(" ", "_")
    output = os.path.join(args.package, f"results_{run_name}.json")
    with open(output, "w") as f:
        json.dump({"machine": platform.node(), "processor": platform.machine(), "detector": args.detector, "model": args.model,
                   "frame_budget_ms": FRAME_BUDGET_MS, "stages_ms": stage_stats, "total_ms": total_stats, "decode_ms": decode_stats,
                   "frames_over_budget": frames_over_budget, "frames": len(all_frames), "flight_fit": fit_report, "shots": shot_rows}, f, indent=1)
    print(f"\nsaved {output}")
    predictor.close()
    if hasattr(detector, "close"):
        detector.close()


if __name__ == "__main__":
    main()
