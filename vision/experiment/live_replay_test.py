#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import queue
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.calibration.court_3d import StereoCourt
from vision.experiment.hit_from_rising_chain import LABEL_WINDOW_AFTER_HIT, MATCH_WINDOW_FRAMES, MODELS_BY_DETECTOR, write_json_atomically
from vision.pipeline.live_mode import (FOLLOWING_BEFORE_HIT, HAILO_MODEL_CHOICES, HAILO_SEARCH_AT_BATCH_1, REPLAY_FALLBACK, SEARCHING, STATES, TRACKING, LiveShotLoop,
                       crop_size_of, distinct_detectors, load_live_detectors)
from vision.robot.esp32_link import DEFAULT_BAUD, DEFAULT_PORT, Esp32Link

CAMERA_FPS = 60
FRAME_BUDGET_MS = 1000 / CAMERA_FPS
WARMUP_CROPS = 5
HANDOFF_FRAMES = 2
CHECKPOINTS_AFTER_HIT = (10, 15, 20)
FRAMES_NEAR_JOIN = 10
CALIBRATION_PACKAGE = "test/pi_test_package"
WAITING_STAGES = ("gray", "aim", "yolo")
END_TO_END_COLUMNS = {"hit_fired": ("fired_on_frame_after_hit", "fired_after_hit_ms"),
                      "first_landing_estimate": ("first_estimate_after_hit_frames", "first_estimate_after_hit_ms"),
                      "final_landing": ("final_after_hit_frames", "final_after_hit_ms")}
TRACKING_STAGES = ("gray", "motion_view", "aim", "yolo", "track", "predict", "finish")


def machine_name():
    return platform.node().split(".")[0].replace(" ", "_")


def run_suffix(args):
    suffix = "" if args.reader_thread else "_no_reader_thread"
    suffix += "" if args.overlap else "_no_overlap"
    suffix += "" if args.seed_tracking else "_replay_at_hit"
    suffix += "_labelled_hits" if args.use_labelled_hits else ""
    return suffix


def output_path(args, kind):
    return os.path.join(args.package, f"live_{kind}_{args.detector}_{args.machine or machine_name()}{run_suffix(args)}.json")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_frame_times(package, manifest):
    with open(os.path.join(package, manifest["frame_times"]), newline="") as f:
        return [float(row["t_s"]) for row in csv.DictReader(f)]


def labelled_hits_fired_late(manifest):
    return {shot["hit_video_frame"] + 3: shot["hit_video_frame"] for shot in manifest["scoring_only"]["shots"]}


class ThreadedReader:
    def __init__(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        self.handoff = queue.Queue(maxsize=HANDOFF_FRAMES)
        self.decode_ms = []
        self.thread = threading.Thread(target=self.read_all, daemon=True)
        self.thread.start()

    def read_all(self):
        while True:
            started = time.perf_counter()
            ok, frame = self.capture.read()
            self.decode_ms.append((time.perf_counter() - started) * 1000)
            self.handoff.put(frame if ok else None)
            if not ok:
                return

    def next_frame(self):
        return self.handoff.get()

    def close(self):
        self.thread.join()
        self.capture.release()


class InlineReader:
    def __init__(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        self.decode_ms = []

    def next_frame(self):
        started = time.perf_counter()
        ok, frame = self.capture.read()
        self.decode_ms.append((time.perf_counter() - started) * 1000)
        return frame if ok else None

    def close(self):
        self.capture.release()


def warm_up(detectors):
    for detector in distinct_detectors(detectors):
        crop_size = crop_size_of(detectors, detector)
        blank = np.zeros((crop_size, crop_size, 3), np.uint8)
        for _ in range(WARMUP_CROPS):
            detector.detect_batch([blank, blank])


def built_loop(args, manifest):
    calibration_manifest = read_json(os.path.join(CALIBRATION_PACKAGE, "manifest.json"))
    court = StereoCourt(read_json(os.path.join(CALIBRATION_PACKAGE, "net_calibration.json")))
    search_model, follow_model = MODELS_BY_DETECTOR[args.detector]
    detectors = load_live_detectors(args.detector, args.model or search_model, args.follow_model or follow_model, args.hailo_models)
    warm_up(detectors)
    forced_hits = labelled_hits_fired_late(manifest) if args.use_labelled_hits else None
    loop = LiveShotLoop(detectors, court, calibration_manifest["right_row_offset_px"], calibration_manifest["terminal_speed_m_s"],
                        calibration_manifest["correction"], overlap=args.overlap, forced_hits=forced_hits,
                        seed_tracking_from_following=args.seed_tracking)
    return loop, detectors, (args.model or search_model, args.follow_model or follow_model)


class LandingSender:
    def __init__(self, link, stream):
        self.link = link
        self.stream = stream
        self.shots_sent = 0
        self.estimates_seen = 0
        self.last_sent_from_frame = None
        self.frames_sent = 0

    def send(self, landing_cm, final):
        self.link.send_landing(*landing_cm, final=final)
        self.frames_sent += 1

    def send_new_estimates(self, estimates):
        for estimate in estimates[self.estimates_seen:]:
            if estimate["landing_cm"] is not None and estimate["landing_from_frame"] != self.last_sent_from_frame:
                self.send(estimate["landing_cm"], final=False)
                self.last_sent_from_frame = estimate["landing_from_frame"]
        self.estimates_seen = len(estimates)

    def send_finished_shots(self, shots):
        while self.shots_sent < len(shots):
            shot = shots[self.shots_sent]
            if self.stream:
                self.send_new_estimates(shot["estimates"])
            if shot["final_landing_cm"] is not None:
                self.send(shot["final_landing_cm"], final=True)
            self.shots_sent += 1
            self.estimates_seen = 0
            self.last_sent_from_frame = None

    def after_push(self, loop):
        if self.stream and loop.tracking is not None:
            self.send_new_estimates(loop.tracking["estimates"])
        self.send_finished_shots(loop.shots)

    def close(self):
        self.link.close()


def landing_sender_from(args):
    return LandingSender(Esp32Link(args.send_uart, args.baud), args.stream) if args.send_uart else None


def run(args):
    manifest = read_json(os.path.join(args.package, "manifest.json"))
    frame_times = read_frame_times(args.package, manifest)
    sender = landing_sender_from(args)
    loop, detectors, models = built_loop(args, manifest)
    video_path = os.path.join(args.package, manifest["video"])
    reader = ThreadedReader(video_path) if args.reader_thread else InlineReader(video_path)
    records, previous_started = [], None
    run_started = time.perf_counter()
    for frame_index, time_s in enumerate(frame_times):
        started = time.perf_counter()
        if previous_started is not None:
            records[-1]["wall_ms"] = round((started - previous_started) * 1000, 3)
        previous_started = started
        stereo_frame = reader.next_frame()
        if stereo_frame is None:
            raise SystemExit(f"could not read video frame {frame_index}")
        fetched_ms = (time.perf_counter() - started) * 1000
        record = loop.push(frame_index, stereo_frame, time_s)
        record["fetch_ms"] = round(fetched_ms, 3)
        records.append(record)
        if sender is not None:
            sender.after_push(loop)
        if record.get("event") == "hit":
            print(f"frame {frame_index}: hit {record['start']}, {record['seeded_frames']} seeded, {record['replayed_frames']} replayed, "
                  f"main {record['main_ms']:.1f} ms", flush=True)
        if record.get("event") == "landing":
            print(f"frame {frame_index}: landing {loop.shots[-1]['final_landing_cm']} (hit estimate {loop.shots[-1]['hit']['hit_estimate']})", flush=True)
    records[-1]["wall_ms"] = round((time.perf_counter() - previous_started) * 1000, 3)
    closing = loop.close()
    run_seconds = time.perf_counter() - run_started
    if sender is not None:
        sender.after_push(loop)
        sender.close()
        print(f"sent {sender.frames_sent} landing frames to {args.send_uart} at {args.baud} baud", flush=True)
    reader.close()
    for detector in distinct_detectors(detectors):
        if hasattr(detector, "close"):
            detector.close()
    settings = {"made": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.node(), "processor": platform.machine(),
                "detector": args.detector, "models": models, "hailo_models": args.hailo_models if args.detector == "hailo" else None,
                "overlap": args.overlap, "reader_thread": args.reader_thread, "seed_tracking": args.seed_tracking, "labelled_hits": args.use_labelled_hits,
                "frames": len(records), "run_seconds": round(run_seconds, 1),
                "decode_ms_median": round(float(np.median(reader.decode_ms)), 3)}
    write_json_atomically(output_path(args, "log"), {"settings": settings, "frames": records, "closing": closing})
    write_json_atomically(output_path(args, "results"), {"settings": settings, "events": loop.events, "shots": loop.shots,
                                                         "replay_fallbacks": loop.replay_fallbacks}, indent=1)
    print(f"{len(records)} frames in {run_seconds:.1f} s, {len(loop.shots)} shots, "
          f"{sum(event['event'] == 'hit' for event in loop.events)} hits, {len(loop.replay_fallbacks)} replay fallbacks "
          f"{[fallback['fired_on_frame'] for fallback in loop.replay_fallbacks]}; saved {output_path(args, 'results')}")


def distance_cm(a, b):
    return None if a is None or b is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def spread(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    values = np.asarray(values, float)
    return {"count": int(len(values)), "mean": float(values.mean()), "min": float(values.min()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def matched_hits(labels, hit_events):
    used, matches = set(), {}
    for label in labels:
        hit = label["hit_video_frame"]
        candidates = [k for k, event in enumerate(hit_events) if k not in used
                      and hit - MATCH_WINDOW_FRAMES <= event["hit_estimate"] <= hit + MATCH_WINDOW_FRAMES
                      and event["fired_at"] <= hit + LABEL_WINDOW_AFTER_HIT]
        if candidates:
            used.add(candidates[0])
            matches[label["shot"]] = candidates[0]
    return matches, used


def join_frames(manifest):
    return [window["first_video_frame"] for window in manifest["windows"][1:]]


def near_join(frame, joins):
    return any(abs(frame - join) <= FRAMES_NEAR_JOIN for join in joins)


def error_counts(errors):
    absolute = [abs(error) for error in errors]
    return {"exact": sum(error == 0 for error in absolute), "within_1": sum(error <= 1 for error in absolute),
            "within_2": sum(error <= 2 for error in absolute), "worse": sum(error > 2 for error in absolute)}


def estimate_on_frame(shot, frame):
    return next((estimate["landing_cm"] for estimate in shot["estimates"] if estimate["frame"] == frame), None)


def reference_checkpoint_misses(manifest):
    return {label["shot"]: {f"miss_hit_plus_{after_hit}_cm": label[f"labelled_hit_kalman_miss_hit_plus_{after_hit}_cm"] for after_hit in CHECKPOINTS_AFTER_HIT}
            for label in manifest["scoring_only"]["shots"]}


def camera_ms(frame_times, later, earlier):
    return None if later is None else round((frame_times[later] - frame_times[earlier]) * 1000, 1)


def shot_rows(manifest, results, frame_times):
    labels = manifest["scoring_only"]["shots"]
    hit_events = [shot["hit"] for shot in results["shots"]]
    matches, used = matched_hits(labels, hit_events)
    references = reference_checkpoint_misses(manifest)
    rows = []
    for label in labels:
        hit = label["hit_video_frame"]
        row = {"shot": label["shot"], "hit_label": hit, "actual_cm": label["actual_cm"],
               "labelled_hit_kalman_landing_cm": label["labelled_hit_kalman_landing_cm"],
               "labelled_hit_kalman_miss_cm": distance_cm(label["labelled_hit_kalman_landing_cm"], label["actual_cm"])}
        if label["shot"] not in matches:
            rows.append(dict(row, missed=True))
            continue
        shot = results["shots"][matches[label["shot"]]]
        event = shot["hit"]
        first_estimate = next((estimate for estimate in shot["estimates"] if estimate["landing_cm"] is not None), None)
        final = shot["final_landing_cm"]
        row.update({
            "missed": False, "hit_estimate": event["hit_estimate"], "error_frames": event["hit_estimate"] - hit,
            "rule_delay_frames": event["fired_at"] - hit, "fired_on_frame_after_hit": event["fired_on_frame"] - hit,
            "fired_after_hit_ms": camera_ms(frame_times, event["fired_on_frame"], hit),
            "first_estimate_after_hit_frames": None if first_estimate is None else first_estimate["arrived_frame"] - hit,
            "first_estimate_after_hit_ms": None if first_estimate is None else camera_ms(frame_times, first_estimate["arrived_frame"], hit),
            "final_after_hit_frames": shot["final_on_frame"] - hit, "final_after_hit_ms": camera_ms(frame_times, shot["final_on_frame"], hit),
            "frames_missing_from_buffer": event.get("frames_missing_from_buffer"), "replayed_frames": event.get("replayed_frames"),
            "tracking_start": event.get("start"), "seeded_frames": event.get("seeded_frames"), "hit_frame_main_ms": event.get("hit_frame_main_ms"),
            "final_landing_cm": final, "miss_cm": distance_cm(final, label["actual_cm"]),
            "signed_x_cm": None if final is None else final[0] - label["actual_cm"][0],
            "signed_y_cm": None if final is None else final[1] - label["actual_cm"][1],
            "moved_from_labelled_hit_cm": distance_cm(final, label["labelled_hit_kalman_landing_cm"]),
            "stereo_points": shot["stereo_points"]})
        row["miss_change_cm"] = None if row["miss_cm"] is None else row["miss_cm"] - row["labelled_hit_kalman_miss_cm"]
        for after_hit in CHECKPOINTS_AFTER_HIT:
            miss = distance_cm(estimate_on_frame(shot, hit + after_hit), label["actual_cm"])
            reference = references[label["shot"]][f"miss_hit_plus_{after_hit}_cm"]
            row[f"miss_hit_plus_{after_hit}_cm"] = miss
            row[f"change_hit_plus_{after_hit}_cm"] = None if miss is None or reference is None else miss - reference
        rows.append(row)
    return rows, used, hit_events


def false_trigger_rows(hit_events, used, joins):
    return [{"fired_on_frame": event["fired_on_frame"], "hit_estimate": event["hit_estimate"],
             "near_join": near_join(event["fired_on_frame"], joins) or near_join(event["hit_estimate"], joins),
             "chain": event.get("chain")} for k, event in enumerate(hit_events) if k not in used]


def state_timing(frames, state, stages):
    chosen = [frame for frame in frames if frame.get("state") == state and frame.get("event") != "hit"]
    if not chosen:
        return None
    calls = [frame.get("detector_call_ms") for frame in chosen]
    return {"frames": len(chosen),
            "stages_ms": {stage: spread([frame["stages_ms"].get(stage, 0.0) for frame in chosen]) for stage in stages},
            "detector_call_ms": spread(calls), "main_ms": spread([frame["main_ms"] for frame in chosen]),
            "wall_ms": spread([frame.get("wall_ms") for frame in chosen]),
            "fetch_ms": spread([frame.get("fetch_ms") for frame in chosen]),
            "over_budget_by_wall": sum((frame.get("wall_ms") or 0) > FRAME_BUDGET_MS for frame in chosen),
            "over_budget_by_main": sum(frame["main_ms"] > FRAME_BUDGET_MS for frame in chosen)}


def hit_frame_timing(frames):
    chosen = [frame for frame in frames if frame.get("event") == "hit"]
    fallbacks = [frame for frame in chosen if frame.get("start") == REPLAY_FALLBACK]
    return {"frames": len(chosen), "main_ms": spread([frame["main_ms"] for frame in chosen]),
            "wall_ms": spread([frame.get("wall_ms") for frame in chosen]),
            "over_budget_by_wall": sum((frame.get("wall_ms") or 0) > FRAME_BUDGET_MS for frame in chosen),
            "seeded_frames": spread([frame.get("seeded_frames") for frame in chosen]),
            "replayed_frames": spread([frame.get("replayed_frames") for frame in chosen]),
            "replay_fallbacks": len(fallbacks), "replay_fallback_frames": [frame["frame"] for frame in fallbacks]}


def landing_summary(rows):
    found = [row for row in rows if not row["missed"]]
    summary = {"final": spread([row["miss_cm"] for row in found]),
               "signed_x_mean": float(np.mean([row["signed_x_cm"] for row in found if row["signed_x_cm"] is not None])) if found else None,
               "signed_y_mean": float(np.mean([row["signed_y_cm"] for row in found if row["signed_y_cm"] is not None])) if found else None,
               "labelled_hit_kalman": spread([row["labelled_hit_kalman_miss_cm"] for row in rows]),
               "moved_from_labelled_hit": spread([row.get("moved_from_labelled_hit_cm") for row in found]),
               "miss_change": spread([row.get("miss_change_cm") for row in found]),
               "shots_without_landing": [row["shot"] for row in rows if row["missed"] or row.get("final_landing_cm") is None]}
    for after_hit in CHECKPOINTS_AFTER_HIT:
        summary[f"hit_plus_{after_hit}"] = spread([row.get(f"miss_hit_plus_{after_hit}_cm") for row in found])
        summary[f"hit_plus_{after_hit}_max_abs_change"] = max([abs(row[f"change_hit_plus_{after_hit}_cm"]) for row in found
                                                              if row.get(f"change_hit_plus_{after_hit}_cm") is not None], default=None)
    return summary


def score(args):
    manifest = read_json(os.path.join(args.package, "manifest.json"))
    frame_times = read_frame_times(args.package, manifest)
    results = read_json(output_path(args, "results"))
    log = read_json(output_path(args, "log"))
    rows, used, hit_events = shot_rows(manifest, results, frame_times)
    false_triggers = false_trigger_rows(hit_events, used, join_frames(manifest))
    found = [row for row in rows if not row["missed"]]
    hit_summary = {"shots": len(rows), "found": len(found), "missed": [row["shot"] for row in rows if row["missed"]],
                   **error_counts([row["error_frames"] for row in found]),
                   "signed_errors": sorted(row["error_frames"] for row in found),
                   "rule_delay_frames": spread([row["rule_delay_frames"] for row in found]),
                   "false_triggers": len(false_triggers), "false_triggers_near_join": sum(row["near_join"] for row in false_triggers),
                   "false_triggers_elsewhere": sum(not row["near_join"] for row in false_triggers)}
    end_to_end = {name: {"frames": spread([row[frames_key] for row in found]), "ms": spread([row[ms_key] for row in found])}
                  for name, (frames_key, ms_key) in END_TO_END_COLUMNS.items()}
    timing = {SEARCHING: state_timing(log["frames"], SEARCHING, WAITING_STAGES),
              FOLLOWING_BEFORE_HIT: state_timing(log["frames"], FOLLOWING_BEFORE_HIT, WAITING_STAGES),
              TRACKING: state_timing(log["frames"], TRACKING, TRACKING_STAGES), "hit_frames": hit_frame_timing(log["frames"]),
              "all_frames": {"main_ms": spread([frame["main_ms"] for frame in log["frames"]]),
                             "wall_ms": spread([frame.get("wall_ms") for frame in log["frames"]])}}
    scored = {"scored_at": time.strftime("%Y-%m-%d %H:%M"), "run_settings": results["settings"], "hits": hit_summary,
              "landing": landing_summary(rows), "end_to_end": end_to_end, "timing": timing, "per_shot": rows,
              "false_triggers": false_triggers}
    write_json_atomically(output_path(args, "score"), scored, indent=1)
    print_summary(scored)
    print(f"\nsaved {output_path(args, 'score')}")


def text(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(scored):
    hits, landing = scored["hits"], scored["landing"]
    print(f"hits: {hits['found']}/{hits['shots']} found, exact {hits['exact']}, within 1 {hits['within_1']}, within 2 {hits['within_2']}, "
          f"worse {hits['worse']}; signed errors {hits['signed_errors']}; false triggers {hits['false_triggers']} "
          f"({hits['false_triggers_near_join']} near a join)")
    final = landing["final"]
    if final:
        print(f"landing miss: mean {final['mean']:.1f}, median {final['median']:.1f}, p90 {final['p90']:.1f}, max {final['max']:.1f} cm; "
              f"signed x {landing['signed_x_mean']:+.1f}, y {landing['signed_y_mean']:+.1f}; labelled-hit Kalman "
              f"{landing['labelled_hit_kalman']['mean']:.1f} / {landing['labelled_hit_kalman']['median']:.1f}; "
              f"moved from it: max {landing['moved_from_labelled_hit']['max']:.2f} cm; without landing {landing['shots_without_landing']}")
    for after_hit in CHECKPOINTS_AFTER_HIT:
        checkpoint = landing[f"hit_plus_{after_hit}"]
        if checkpoint:
            print(f"  hit +{after_hit}: mean {checkpoint['mean']:.1f}, median {checkpoint['median']:.1f} cm ({checkpoint['count']} shots); "
                  f"max change vs labelled-hit run {text(landing[f'hit_plus_{after_hit}_max_abs_change'], 2)} cm")
    print(f"\n  {'shot':>4}{'err':>5}{'fired':>6}{'1st est':>8}{'final':>6}{'miss':>7}{'labelled':>9}{'moved':>7}{'+10':>7}{'+15':>7}{'+20':>7}")
    for row in scored["per_shot"]:
        if row["missed"]:
            print(f"  {row['shot']:>4}  missed")
            continue
        print(f"  {row['shot']:>4}{row['error_frames']:>+5}{row['fired_on_frame_after_hit']:>+6}{text(row['first_estimate_after_hit_frames'], 0):>8}"
              f"{row['final_after_hit_frames']:>+6}{text(row['miss_cm']):>7}{text(row['labelled_hit_kalman_miss_cm']):>9}"
              f"{text(row['moved_from_labelled_hit_cm'], 2):>7}{text(row['miss_hit_plus_10_cm']):>7}{text(row['miss_hit_plus_15_cm']):>7}"
              f"{text(row['miss_hit_plus_20_cm']):>7}")
    print(f"\n  {'state':<11}{'frames':>7}  {'stage medians ms':<52}{'call':>6}{'main':>7}{'p95':>7}{'wall':>7}{'p95':>7}{'>16.7 wall/main':>17}")
    for state in STATES:
        timing = scored["timing"][state]
        if timing is None:
            continue
        stages = " ".join(f"{stage} {stats['median']:.1f}" for stage, stats in timing["stages_ms"].items() if stats)
        call = timing["detector_call_ms"]["median"] if timing["detector_call_ms"] else None
        print(f"  {state:<11}{timing['frames']:>7}  {stages:<52}{text(call):>6}{timing['main_ms']['median']:>7.1f}{timing['main_ms']['p95']:>7.1f}"
              f"{timing['wall_ms']['median']:>7.1f}{timing['wall_ms']['p95']:>7.1f}{timing['over_budget_by_wall']:>9}/{timing['over_budget_by_main']}")
    hit_frames = scored["timing"]["hit_frames"]
    if hit_frames["main_ms"]:
        print(f"  hit frames: {hit_frames['frames']}, main median {hit_frames['main_ms']['median']:.1f} ms (max {hit_frames['main_ms']['max']:.1f}), "
              f"wall median {text(hit_frames['wall_ms']['median'] if hit_frames['wall_ms'] else None)} ms, "
              f"over 16.7 by wall {hit_frames['over_budget_by_wall']}; seeded median {text(hit_frames['seeded_frames']['median'] if hit_frames['seeded_frames'] else None, 0)}, "
              f"replayed median {text(hit_frames['replayed_frames']['median'] if hit_frames['replayed_frames'] else None, 0)} frames; "
              f"replay fallbacks {hit_frames['replay_fallbacks']} {hit_frames['replay_fallback_frames']}")
    for name, delay in scored["end_to_end"].items():
        if delay["frames"]:
            print(f"  true hit -> {name}: median {delay['frames']['median']:.0f} frames ({delay['frames']['min']:.0f} to "
                  f"{delay['frames']['max']:.0f}), median {delay['ms']['median']:.0f} ms camera time")


def main():
    parser = argparse.ArgumentParser(description="Replay the chained test video through the live loop (run), then score it against the labels (score).")
    parser.add_argument("command", choices=("run", "score"))
    parser.add_argument("--package", default="test/pi_live_test_package")
    parser.add_argument("--detector", choices=tuple(MODELS_BY_DETECTOR), default="onnx")
    parser.add_argument("--model", help="640 gray-stack model (default by detector)")
    parser.add_argument("--follow-model", help="320 gray-stack model (default by detector)")
    parser.add_argument("--hailo-models", choices=HAILO_MODEL_CHOICES, default=HAILO_SEARCH_AT_BATCH_1,
                        help="search-batch-1: the 640 .hef at batch 1 (searching) and batch 2 (tracking), the 320 .hef at batch 2 (following and tracking); "
                             "shared-batch-2: batch 2 only, searching pads single crops")
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reader-thread", action=argparse.BooleanOptionalAction, default=True,
                        help="decode the video on its own thread, as a camera would deliver frames")
    parser.add_argument("--seed-tracking", action=argparse.BooleanOptionalAction, default=True,
                        help="follow with both eyes before the hit and start tracking from those detections; "
                             "--no-seed-tracking: left eye only, replay the buffered frames through tracking at the hit")
    parser.add_argument("--use-labelled-hits", action="store_true",
                        help="debug only: start tracking at the labelled hit, fired when frame hit + 3 arrives")
    parser.add_argument("--machine", help="score: the machine name in the file names (default this machine)")
    parser.add_argument("--send-uart", nargs="?", const=DEFAULT_PORT, metavar="PORT",
                        help=f"run: send each shot's final landing to the ESP32 as it happens (default port {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--stream", action="store_true", help="with --send-uart: also send every in-flight estimate as it is ready")
    args = parser.parse_args()
    {"run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
