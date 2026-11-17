#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.calibration.court_3d import StereoCourt
from vision.detection.crop_tracking import CROP_SIZE, FOLLOW_CROP_SIZE
from vision.detection.detectors import DETECTOR_KINDS, INPUT_KINDS, MOTION_VIEW_INPUT, STEREO_BATCH, load_detector
from vision.flight.flight_model import FLIGHT_ENGINE
from vision.landing_prediction.landing_worker import LandingPredictor
from vision.pipeline.shot_pipeline import FIT_EVERY_FRAMES, LANDING_BY_FIT, LANDING_BY_KALMAN, LANDING_METHODS, SEARCH_WITH_FOLLOW_CHOICES, STAGES, ShotPipeline

CAMERA_FPS = 60
FRAME_BUDGET_MS = 1000 / CAMERA_FPS
WARMUP_CROPS = 5
CHECKPOINTS_AFTER_HIT = (10, 15, 20)


def warm_up(detector, crop_size=CROP_SIZE):
    blank = np.zeros((crop_size, crop_size, 3), np.uint8)
    for _ in range(WARMUP_CROPS):
        detector.detect_batch([blank, blank])


def run_shot(package_dir, entry, detector, court, manifest, predictor, fit_every, follow_detector=None, search_with_follow_detector=None,
             overlap_detection=False, landing_method=LANDING_BY_FIT):
    pipeline = ShotPipeline(detector, court, manifest["right_row_offset_px"], manifest["terminal_speed_m_s"],
                            manifest["correction"], entry["hit_frame"], entry["end_frame"] + manifest["frames_after_hit"],
                            fit_every, predictor, follow_detector, search_with_follow_detector, overlap_detection, landing_method)
    clip = cv2.VideoCapture(os.path.join(package_dir, "clips", entry["clip"]))
    processed, decode_ms, frame_wall_ms = [], [], []
    first_prediction_frame = None
    frame_started_at = None
    for offset, time_s in enumerate(entry["frame_times_s"]):
        started = time.perf_counter()
        if frame_started_at is not None:
            frame_wall_ms.append((started - frame_started_at) * 1000)
        ok, stereo_frame = clip.read()
        decode_ms.append((time.perf_counter() - started) * 1000)
        if not ok:
            raise SystemExit(f"{entry['clip']}: could not read frame {offset}")
        result = pipeline.push(entry["first_frame"] + offset, stereo_frame, time_s)
        frame_started_at = started if result is not None or pipeline.pending_detection is not None else None
        if result is not None:
            processed.append(result)
    drain_started = time.perf_counter()
    if frame_started_at is not None:
        frame_wall_ms.append((drain_started - frame_started_at) * 1000)
    drained = pipeline.drain()
    if drained is not None:
        processed.append(drained)
        frame_wall_ms.append((time.perf_counter() - drain_started) * 1000)
    first_prediction_frame = next((result["frame"] for result in processed if result["predicted"]), None)
    final = pipeline.finish()
    landing = None if final is None else [round(float(v), 1) for v in final]
    return processed, decode_ms, frame_wall_ms, landing, first_prediction_frame, pipeline


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


def print_timing_row(name, stats, note=""):
    print(f"  {name:<14}{stats['mean']:9.2f}{stats['median']:9.2f}{stats['p95']:9.2f}{stats['max']:9.2f}{note}")


def print_report(stage_stats, total_stats, decode_stats, shot_rows, frames_over_budget, frame_count, predict_on_frames, fit_report, overlap_report):
    print(f"\nper processed stereo frame, {frame_count} frames over {len(shot_rows)} shots (budget {FRAME_BUDGET_MS:.1f} ms at {CAMERA_FPS} fps)")
    print(f"  {'stage':<14}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}  ms")
    for stage in STAGES:
        s = stage_stats[stage]
        print(f"  {stage:<14}{s['mean']:9.2f}{s['median']:9.2f}{s['p95']:9.2f}{s['max']:9.2f}")
    print(f"  {'TOTAL':<14}{total_stats['mean']:9.2f}{total_stats['median']:9.2f}{total_stats['p95']:9.2f}{total_stats['max']:9.2f}")
    print(f"  {'(video decode)':<14}{decode_stats['mean']:9.2f}{decode_stats['median']:9.2f}{decode_stats['p95']:9.2f}{decode_stats['max']:9.2f}  not in TOTAL")
    print_timing_row("WALL", overlap_report["frame_wall_ms"], "  one frame's start to the next, decode included")
    if overlap_report["detector_call_ms"]:
        print_timing_row("(detector)", overlap_report["detector_call_ms"], "  detector busy per call, overlaps the stages")
    print_timing_row("(estimate)", overlap_report["ready_after_inputs_ms"], "  newest needed frame in -> its estimate ready")
    print(f"  detection overlapped with the CPU stages: {'yes' if overlap_report['overlap'] else 'no'}")
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
    parser.add_argument("--package", default="test/pi_test_package")
    parser.add_argument("--detector", choices=DETECTOR_KINDS, default="ncnn")
    parser.add_argument("--model", required=True, help="bird_yolo11n_v1_640_ncnn_model folder, bird_yolo11n_v1.pt, or the .hef file")
    parser.add_argument("--follow-model", help=f"same detector at {FOLLOW_CROP_SIZE} x {FOLLOW_CROP_SIZE}, used while following the bird; --model still searches at {CROP_SIZE}")
    parser.add_argument("--search-with-follow-model", choices=SEARCH_WITH_FOLLOW_CHOICES,
                        help=f"search with --follow-model too: always, or once-seen (only after the first detection; the first full-frame search stays at {CROP_SIZE}); the blob search window stays {CROP_SIZE}")
    parser.add_argument("--hailo-input-bgr", action="store_true", help="feed the Hailo BGR instead of RGB (match how the HEF was calibrated)")
    parser.add_argument("--hailo-batch", type=int, default=STEREO_BATCH, help="crops per Hailo call; 2 runs both eyes at once, 1 one eye at a time")
    parser.add_argument("--shots", type=int, nargs="+", help="only these shot numbers")
    parser.add_argument("--input", choices=INPUT_KINDS, default=MOTION_VIEW_INPUT, help="gray-stack: the model builds the motion view itself from 3 gray crops")
    parser.add_argument("--fit-every", type=int, default=FIT_EVERY_FRAMES, help="start a flight fit every N frames")
    parser.add_argument("--fit-background", action="store_true", help="run the flight fit in a separate process on another core instead of inside the frame loop")
    parser.add_argument("--overlap", action="store_true",
                        help="start the detector on frame t+1 and run tracking, the flight fit and the next gray conversion while it works; same landings, one call later")
    parser.add_argument("--landing-method", choices=LANDING_METHODS, default=LANDING_BY_FIT,
                        help="fit: least-squares flight fit + correction (default); kalman: C++ Kalman filter + its correction (vision/landing_prediction/kalman_landing_settings.json)")
    parser.add_argument("--landing-tuning", help="landing_experiment_*.json in the package tuned on this model's tracks (terminal speed + correction)")
    args = parser.parse_args()
    if args.search_with_follow_model and not args.follow_model:
        parser.error("--search-with-follow-model needs --follow-model")

    with open(os.path.join(args.package, "manifest.json")) as f:
        manifest = json.load(f)
    if args.landing_tuning:
        apply_landing_tuning(manifest, os.path.join(args.package, args.landing_tuning))
    with open(os.path.join(args.package, "net_calibration.json")) as f:
        court = StereoCourt(json.load(f))
    predictor = LandingPredictor(manifest["terminal_speed_m_s"], manifest["correction"], background=args.fit_background)
    detector = load_detector(args.detector, args.model, args.hailo_input_bgr, args.input, args.hailo_batch)
    follow_detector = None
    if args.follow_model:
        follow_detector = load_detector(args.detector, args.follow_model, args.hailo_input_bgr, args.input, args.hailo_batch, on_hailo_device_of=detector)
    warm_up(detector)
    if follow_detector is not None:
        warm_up(follow_detector, FOLLOW_CROP_SIZE)

    all_frames, all_decode, all_frame_wall, shot_rows, all_fit_ms = [], [], [], [], []
    fits = waited = 0
    for entry in manifest["shots"]:
        if args.shots and entry["shot"] not in args.shots:
            continue
        processed, decode_ms, frame_wall_ms, landing, first_prediction_frame, pipeline = run_shot(
            args.package, entry, detector, court, manifest, predictor, args.fit_every, follow_detector, args.search_with_follow_model, args.overlap, args.landing_method)
        all_frame_wall.extend(frame_wall_ms)
        all_fit_ms.extend(pipeline.fit_ms)
        fits += pipeline.fits_started
        waited += pipeline.fits_waiting_for_worker
        totals = [sum(frame["stages_ms"].values()) for frame in processed]
        all_frames.extend(processed)
        all_decode.extend(decode_ms)
        shot_rows.append({
            "shot": entry["shot"], "frames": len(processed), "stereo_points": len(pipeline.stereo),
            "follow_crop_frames": sum(frame["crop_size"] == FOLLOW_CROP_SIZE for frame in processed) if follow_detector else 0,
            "mean_ms": float(np.mean(totals)), "wall_mean_ms": float(np.mean(frame_wall_ms)), "p95_ms": float(np.percentile(totals, 95)),
            "first_prediction_after_hit": None if first_prediction_frame is None else first_prediction_frame - entry["hit_frame"],
            "landing_cm": landing, "miss_cm": distance_cm(landing, entry["actual_cm"]),
            "vs_mac_cm": distance_cm(landing, entry["mac_corrected_cm"]),
            "checkpoint_miss_cm": {str(k): distance_cm(estimate_at(processed, entry["hit_frame"] + k), entry["actual_cm"]) for k in CHECKPOINTS_AFTER_HIT},
        })
        print(f"shot {entry['shot']:>3}: {len(processed)} frames, {np.mean(totals):.1f} ms per frame in stages, {np.mean(frame_wall_ms):.1f} ms wall, landing {landing}")

    stage_stats = {stage: summary([frame["stages_ms"].get(stage, 0.0) for frame in all_frames]) for stage in STAGES}
    totals = [sum(frame["stages_ms"].values()) for frame in all_frames]
    total_stats = summary(totals)
    decode_stats = summary(all_decode)
    frames_over_budget = int(sum(total > FRAME_BUDGET_MS for total in totals))
    predict_on_frames = [frame["stages_ms"]["predict"] for frame in all_frames if frame["predicted"]]
    fit_report = {"mode": LANDING_BY_KALMAN if args.landing_method == LANDING_BY_KALMAN else "background" if args.fit_background else "inline", "engine": FLIGHT_ENGINE, "every": args.fit_every, "fits": fits, "waited": waited,
                  "fit_ms": summary(all_fit_ms) if all_fit_ms else None}
    detector_calls = [frame["detector_call_ms"] for frame in all_frames if frame["detector_call_ms"] is not None]
    overlap_report = {"overlap": args.overlap, "frame_wall_ms": summary(all_frame_wall),
                      "detector_call_ms": summary(detector_calls) if detector_calls else None,
                      "ready_after_inputs_ms": summary([frame["ready_after_inputs_ms"] for frame in all_frames])}
    print_report(stage_stats, total_stats, decode_stats, shot_rows, frames_over_budget, len(all_frames), predict_on_frames, fit_report, overlap_report)
    yolo_by_crop_size = {str(size): summary(yolo) for size in (CROP_SIZE, FOLLOW_CROP_SIZE)
                         for yolo in [[frame["stages_ms"]["yolo"] for frame in all_frames if frame["crop_size"] == size]] if yolo}
    for size, yolo in yolo_by_crop_size.items():
        frames_at_size = sum(frame["crop_size"] == int(size) for frame in all_frames)
        print(f"  yolo at {size} x {size}: {yolo['mean']:.2f} ms mean, {yolo['median']:.2f} median per frame (both eyes), {frames_at_size}/{len(all_frames)} frames")

    model_name = os.path.splitext(os.path.basename(os.path.normpath(args.model)))[0]
    fit_name = (f"background{args.fit_every}" if args.fit_background else f"inline{args.fit_every}") + ("_cpp" if FLIGHT_ENGINE == "c++" else "")
    fit_name += f"_batch{args.hailo_batch}" if args.detector == "hailo" else ""
    fit_name += f"_follow{FOLLOW_CROP_SIZE}" if follow_detector else ""
    fit_name += f"_search_{args.search_with_follow_model}" if args.search_with_follow_model else ""
    fit_name += "_overlap" if args.overlap else ""
    fit_name += "_kalman" if args.landing_method == LANDING_BY_KALMAN else ""
    run_name = f"{platform.node().split('.')[0]}_{args.detector}_{model_name}_{fit_name}".replace(" ", "_")
    output = os.path.join(args.package, f"results_{run_name}.json")
    with open(output, "w") as f:
        json.dump({"machine": platform.node(), "processor": platform.machine(), "detector": args.detector, "model": args.model,
                   "frame_budget_ms": FRAME_BUDGET_MS, "stages_ms": stage_stats, "total_ms": total_stats, "decode_ms": decode_stats, **overlap_report,
                   "frames_over_budget": frames_over_budget, "frames": len(all_frames), "yolo_ms_by_crop_size": yolo_by_crop_size, "flight_fit": fit_report, "landing_method": args.landing_method, "shots": shot_rows}, f, indent=1)
    print(f"\nsaved {output}")
    predictor.close()
    for loaded in (follow_detector, detector):
        if hasattr(loaded, "close"):
            loaded.close()


if __name__ == "__main__":
    main()
