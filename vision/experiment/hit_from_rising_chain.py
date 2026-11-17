#!/usr/bin/env python3
import argparse
import collections
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vision.detection.crop_tracking import CROP_SIZE, FOLLOW_CROP_SIZE
from vision.detection.detectors import GRAY_STACK_INPUT, load_detector
from vision.tracking.hit_detection import WaitingDetector, best_chain_ending_at, chain_fires, hit_estimate, hit_event, milliseconds_since, REFRACTORY_FRAMES

CAMERA_FPS = 60
MARGIN_AROUND_SHOTS = 150
FRAMES_AFTER_LANDING_TO_EXCLUDE = 40
WARMUP_CROPS = 5
CACHE_FOLDER = "hit_detection_cache"
RESULT_NAME = "hit_detection_baseline_train.json"
MATCH_WINDOW_FRAMES = 20
LABEL_WINDOW_AFTER_HIT = 45
TAKES_TIMED_ALONE = ("take_018",)
MODELS_BY_DETECTOR = {
    "onnx": ("models/bird_yolo11n_v1_gray_stack.onnx", "models/bird_yolo11n_v1_gray_stack_320.onnx"),
    "hailo": ("models/bird_yolo11n_v1_gray_stack.hef", "models/bird_yolo11n_v1_gray_stack_320.hef"),
}
PACKAGE_TAKES = ("take_000", "take_006", "take_011")
MANIFEST_NAME = "manifest.json"
CLIP_FOLDER = "clips"


def read_shots(session):
    with open(os.path.join(session, "shots.csv"), newline="") as f:
        return [{"shot": int(row["shot"]), "take": row["take"], "split": row["split"], "start": int(row["start_frame"]),
                 "end": int(row["end_frame"]), "hit": int(row["hit_frame"])} for row in csv.DictReader(f)]


def shots_by_take(shots):
    grouped = collections.defaultdict(list)
    for shot in shots:
        grouped[shot["take"]].append(shot)
    return {take: sorted(group, key=lambda shot: shot["start"]) for take, group in grouped.items()}


def excluded_test_ranges(take_shots):
    return [(shot["start"], shot["end"] + FRAMES_AFTER_LANDING_TO_EXCLUDE) for shot in take_shots if shot["split"] == "test"]


def processed_ranges(take_shots, frame_count):
    train = [shot for shot in take_shots if shot["split"] == "train"]
    first = max(0, train[0]["start"] - MARGIN_AROUND_SHOTS)
    last = min(frame_count - 1, train[-1]["end"] + MARGIN_AROUND_SHOTS)
    ranges, cursor = [], first
    for excluded_first, excluded_last in excluded_test_ranges(take_shots):
        if excluded_last < first or excluded_first > last:
            continue
        if excluded_first > cursor:
            ranges.append((cursor, excluded_first - 1))
        cursor = max(cursor, excluded_last + 1)
    if cursor <= last:
        ranges.append((cursor, last))
    return ranges


def plain_json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomically(path, value, **dump_options):
    temporary_path = path + ".writing"
    with open(temporary_path, "w") as f:
        json.dump(value, f, default=plain_json_value, **dump_options)
    os.replace(temporary_path, path)


def warm_up(detector, crop_size):
    blank = np.zeros((crop_size, crop_size, 3), np.uint8)
    for _ in range(WARMUP_CROPS):
        detector.detect(blank)


def add_cycle_timing(entry, gray_ms, decode_ms, frame_wall_ms):
    entry["gray_ms"] = round(float(gray_ms), 2)
    entry["decode_ms"] = round(decode_ms, 2)
    entry["main_ms"] = round(entry["gray_ms"] + entry["aim_ms"] + entry["yolo_ms"], 2)
    entry["frame_wall_ms"] = round(frame_wall_ms, 2)


def store_finished(finished, cycle_timing, per_frame):
    if finished is not None:
        finished_frame, entry = finished
        add_cycle_timing(entry, *cycle_timing.pop(finished_frame))
        per_frame[finished_frame] = entry


def detect_range(capture, first, last, waiting, per_frame):
    waiting.reset()
    cycle_timing = {}
    for frame_index in range(first, last + 1):
        cycle_started = time.perf_counter()
        ok, stereo_frame = capture.read()
        if not ok:
            break
        decode_ms = milliseconds_since(cycle_started)
        gray_ms, started_middle, finished = waiting.step(stereo_frame, frame_index)
        if started_middle is not None:
            cycle_timing[started_middle] = (gray_ms, decode_ms, milliseconds_since(cycle_started))
        store_finished(finished, cycle_timing, per_frame)
    store_finished(waiting.drain(), cycle_timing, per_frame)


def detect_take(session, take, take_shots, waiting):
    capture = cv2.VideoCapture(os.path.join(session, take + ".avi"))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ranges = processed_ranges(take_shots, frame_count)
    per_frame = {}
    for first, last in ranges:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        detect_range(capture, first, last, waiting, per_frame)
    return {"take": take, "frame_count": frame_count, "ranges": ranges, "frames": per_frame}


def detect_package_take(package, take_entry, waiting):
    per_frame = {}
    for clip in take_entry["clips"]:
        capture = cv2.VideoCapture(os.path.join(package, clip["file"]))
        detect_range(capture, clip["first_frame"], clip["last_frame"], waiting, per_frame)
        capture.release()
    ranges = [[clip["first_frame"], clip["last_frame"]] for clip in take_entry["clips"]]
    return {"take": take_entry["take"], "frame_count": take_entry["frame_count"], "ranges": ranges, "frames": per_frame}


def read_manifest(package):
    with open(os.path.join(package, MANIFEST_NAME)) as f:
        return json.load(f)


def cache_folder(args):
    root = args.package or args.session
    folder = CACHE_FOLDER if args.detector == "onnx" and not args.package else f"{CACHE_FOLDER}_{args.detector}"
    return os.path.join(root, folder + overlap_suffix(args))


def overlap_suffix(args):
    return "_overlap" if args.overlap else ""


def models_for(args):
    search_model, follow_model = MODELS_BY_DETECTOR[args.detector]
    return args.model or search_model, args.follow_model or follow_model


def load_waiting_detectors(detector_kind, search_model, follow_model):
    search_detector = load_detector(detector_kind, search_model, input_kind=GRAY_STACK_INPUT, hailo_batch_size=1)
    follow_detector = load_detector(detector_kind, follow_model, input_kind=GRAY_STACK_INPUT, hailo_batch_size=1,
                                    on_hailo_device_of=search_detector if detector_kind == "hailo" else None)
    return search_detector, follow_detector


def close_detectors(*detectors):
    for detector in detectors:
        if hasattr(detector, "close"):
            detector.close()


def run_detection(args):
    search_model, follow_model = models_for(args)
    if args.package:
        manifest = read_manifest(args.package)
        take_names = args.takes or sorted(manifest["takes"])
    else:
        grouped = shots_by_take(read_shots(args.session))
        take_names = args.takes or sorted(grouped)
    cache = cache_folder(args)
    os.makedirs(cache, exist_ok=True)
    search_detector, follow_detector = load_waiting_detectors(args.detector, search_model, follow_model)
    warm_up(search_detector, CROP_SIZE)
    warm_up(follow_detector, FOLLOW_CROP_SIZE)
    waiting = WaitingDetector(search_detector, follow_detector, args.overlap)
    for take in take_names:
        path = os.path.join(cache, take + ".json")
        if os.path.exists(path) and not args.redo:
            continue
        started = time.perf_counter()
        if args.package:
            result = detect_package_take(args.package, manifest["takes"][take], waiting)
        else:
            result = detect_take(args.session, take, grouped[take], waiting)
        result.update({"detector": args.detector, "model": search_model, "follow_model": follow_model,
                       "machine": platform.node(), "overlap": args.overlap, "run_seconds": round(time.perf_counter() - started, 1)})
        write_json_atomically(path, result)
        print(f"{take}: {len(result['frames'])} frames in {result['run_seconds']} s, "
              f"{result['run_seconds'] * 1000 / max(1, len(result['frames'])):.1f} ms per frame", flush=True)
    close_detectors(follow_detector, search_detector)


def load_cache(cache):
    takes = {}
    for name in sorted(os.listdir(cache)):
        with open(os.path.join(cache, name)) as f:
            take = json.load(f)
        take["frames"] = {int(frame): entry for frame, entry in take["frames"].items()}
        takes[take["take"]] = take
    return takes


def chains_per_frame(take, min_detection_confidence):
    found, chains = {}, {}
    for frame in sorted(take["frames"]):
        detections = [(x, y, c) for x, y, c in take["frames"][frame]["detections"] if c >= min_detection_confidence]
        found[frame] = detections
        if detections:
            chain = best_chain_ending_at(found, frame)
            if len(chain) >= 2:
                chains[frame] = chain
    return chains


def hit_estimate_for_rule(chain, rule):
    if rule["estimate"] == "first-point":
        return hit_estimate(chain, rule["offset"])
    used = chain[:rule["fit_points"]]
    frames = np.array([point[0] for point in used], float)
    rows = np.array([point[2] for point in used], float)
    slope, intercept = np.polyfit(frames, rows, 1)
    if slope >= 0:
        return hit_estimate(chain, rule["offset"])
    return min(chain[0][0], int(round((rule["start_row"] - intercept) / slope))) - rule["offset"]


def fire_events(chains, rule):
    events, quiet_until = [], -1
    for frame in sorted(chains):
        if frame <= quiet_until:
            continue
        chain = chains[frame]
        if not chain_fires(chain, rule):
            continue
        events.append(hit_event(frame, chain, hit_estimate_for_rule(chain, rule)))
        quiet_until = frame + REFRACTORY_FRAMES
    return events


def in_ranges(frame, ranges):
    return any(first <= frame <= last for first, last in ranges)


def shot_window(shot):
    return shot["start"], shot["end"] + FRAMES_AFTER_LANDING_TO_EXCLUDE


def flight_window(shot):
    return shot["hit"] - MATCH_WINDOW_FRAMES, shot["end"] + FRAMES_AFTER_LANDING_TO_EXCLUDE


def score_take(take, take_shots, rule, chains):
    events = fire_events(chains, rule)
    train = [shot for shot in take_shots if shot["split"] == "train"]
    per_shot, used = [], set()
    for shot in train:
        candidates = [k for k, event in enumerate(events)
                      if shot["hit"] - MATCH_WINDOW_FRAMES <= event["hit_estimate"] <= shot["hit"] + MATCH_WINDOW_FRAMES
                      and event["fired_at"] <= shot["hit"] + LABEL_WINDOW_AFTER_HIT]
        if candidates:
            k = candidates[0]
            used.add(k)
            event = events[k]
            per_shot.append({"shot": shot["shot"], "take": take["take"], "hit_label": shot["hit"], "hit_estimate": event["hit_estimate"],
                             "error_frames": event["hit_estimate"] - shot["hit"], "delay_frames": event["fired_at"] - shot["hit"],
                             "chain_first_frame": event["chain_first_frame"], "chain_points": len(event["chain"]),
                             "chain": event["chain"]})
        else:
            per_shot.append({"shot": shot["shot"], "take": take["take"], "hit_label": shot["hit"], "missed": True})
    false_triggers = []
    for k, event in enumerate(events):
        if k in used:
            continue
        inside_shot = next((shot for shot in train if shot_window(shot)[0] <= event["fired_at"] <= shot_window(shot)[1]), None)
        false_triggers.append({"take": take["take"], "fired_at": event["fired_at"], "hit_estimate": event["hit_estimate"],
                               "chain_points": len(event["chain"]), "chain": event["chain"],
                               "where": "gap" if inside_shot is None else
                               f"shot {inside_shot['shot']} window, {event['hit_estimate'] - inside_shot['hit']:+d} from hit"})
    return per_shot, false_triggers


def gap_frames(take, take_shots):
    train_windows = [shot_window(shot) for shot in take_shots if shot["split"] == "train"]
    return sum(1 for frame in take["frames"] if not in_ranges(frame, train_windows))


def no_flight_frames(take, take_shots):
    flights = [flight_window(shot) for shot in take_shots if shot["split"] == "train"]
    return sum(1 for frame in take["frames"] if not in_ranges(frame, flights))


def summarise(per_shot, false_triggers, gap_frame_count, no_flight_frame_count):
    found = [entry for entry in per_shot if not entry.get("missed")]
    errors = [abs(entry["error_frames"]) for entry in found]
    gap_minutes = gap_frame_count / CAMERA_FPS / 60
    gap_false = [trigger for trigger in false_triggers if trigger["where"] == "gap"]
    return {
        "shots": len(per_shot), "found": len(found), "missed": len(per_shot) - len(found),
        "exact": sum(error == 0 for error in errors), "within_1": sum(error <= 1 for error in errors),
        "within_2": sum(error <= 2 for error in errors), "worse_than_2": sum(error > 2 for error in errors),
        "mean_abs_error_frames": round(statistics.mean(errors), 2) if errors else None,
        "median_abs_error_frames": statistics.median(errors) if errors else None,
        "mean_signed_error_frames": round(statistics.mean(entry["error_frames"] for entry in found), 2) if found else None,
        "delay_frames_median": statistics.median(entry["delay_frames"] for entry in found) if found else None,
        "delay_frames_max": max(entry["delay_frames"] for entry in found) if found else None,
        "false_triggers_total": len(false_triggers), "false_triggers_in_gaps": len(gap_false),
        "false_triggers_in_shot_windows": len(false_triggers) - len(gap_false),
        "gap_minutes": round(gap_minutes, 2),
        "false_triggers_per_gap_minute": round(len(gap_false) / gap_minutes, 3) if gap_minutes else None,
        "no_flight_minutes": round(no_flight_frame_count / CAMERA_FPS / 60, 2),
        "false_triggers_per_no_flight_minute": round(len(false_triggers) / (no_flight_frame_count / CAMERA_FPS / 60), 3),
    }


def score_every_take(takes, grouped, rules, chains_by_confidence):
    table = []
    for rule in rules:
        per_take = {}
        for name, take in takes.items():
            shot_rows, triggers = score_take(take, grouped[name], rule, chains_by_confidence[rule["min_detection_confidence"]][name])
            per_take[name] = (shot_rows, triggers, gap_frames(take, grouped[name]), no_flight_frames(take, grouped[name]))
        table.append(per_take)
    return table


def combined(per_take, names):
    per_shot, false_triggers, gap_count, no_flight_count = [], [], 0, 0
    for name in names:
        shot_rows, triggers, gaps, no_flight = per_take[name]
        per_shot += shot_rows
        false_triggers += triggers
        gap_count += gaps
        no_flight_count += no_flight
    return summarise(per_shot, false_triggers, gap_count, no_flight_count), per_shot, false_triggers


def best_rule_index(table, names):
    return min(range(len(table)), key=lambda k: (rule_score(combined(table[k], names)[0]), k))


def leave_one_take_out(table, rules, names):
    picked, held_out_results = {}, {}
    for held_out in names:
        k = best_rule_index(table, [name for name in names if name != held_out])
        picked[held_out] = rules[k]
        held_out_results[held_out] = table[k][held_out]
    summary, per_shot, false_triggers = combined(held_out_results, names)
    return summary, picked, per_shot, false_triggers


def candidate_rules():
    for min_points in (3, 4, 5, 6):
        for min_rise in (20, 40, 60, 100):
            for min_confidence in (0.25, 0.4, 0.5):
                for min_start_row in (0, 350, 420):
                    base = {"min_points": min_points, "min_rise_px": min_rise, "min_detection_confidence": min_confidence,
                            "max_span_frames": 20, "min_start_row": min_start_row}
                    for offset in range(-2, 4):
                        yield dict(base, estimate="first-point", offset=offset)


def rule_score(summary):
    return (summary["missed"] + summary["false_triggers_total"], summary["worse_than_2"], -summary["within_1"],
            summary["mean_abs_error_frames"] or 99, summary["delay_frames_median"] or 99)


def idle_timing(takes):
    search = [entry for take in takes.values() for entry in take["frames"].values() if entry["mode"] == "search"]
    follow = [entry for take in takes.values() for entry in take["frames"].values() if entry["mode"] != "search"]

    def stage_medians(entries):
        stages = [stage for stage in ("gray_ms", "aim_ms", "yolo_ms", "detector_call_ms", "main_ms", "decode_ms", "frame_wall_ms")
                  if entries and all(stage in entry for entry in entries)]
        return {stage: round(statistics.median(entry[stage] for entry in entries), 2) for stage in stages} | {
            "total_ms_median": round(statistics.median(entry["gray_ms"] + entry["aim_ms"] + entry["yolo_ms"] for entry in entries), 2),
            "total_ms_mean": round(statistics.mean(entry["gray_ms"] + entry["aim_ms"] + entry["yolo_ms"] for entry in entries), 2),
            "frames": len(entries)}
    return {"search_640_full_frame_blobs": stage_medians(search), "following_320": stage_medians(follow)}


def run_evaluation(args):
    if args.package:
        evaluate_package(args)
        return
    shots = read_shots(args.session)
    grouped = shots_by_take(shots)
    takes = load_cache(cache_folder(args))
    rules = list(candidate_rules()) + list(extrapolation_rules())
    chains_by_confidence = {confidence: {name: chains_per_frame(take, confidence) for name, take in takes.items()}
                            for confidence in sorted({rule["min_detection_confidence"] for rule in rules})}
    table = score_every_take(takes, grouped, rules, chains_by_confidence)
    names = sorted(takes)
    scored = sorted((rule_score(combined(table[k], names)[0]), k) for k in range(len(rules)))
    best_index = scored[0][1]
    best_rule = rules[best_index]
    summary, per_shot, false_triggers = combined(table[best_index], names)
    processed = sum(len(take["frames"]) for take in takes.values())
    cross_summary, picked, cross_per_shot, cross_false = leave_one_take_out(table, rules, names)
    leave_one_take_out_result = {"summary": cross_summary, "rule_picked_per_held_out_take": picked,
                                 "misses_and_errors_over_2": [entry for entry in cross_per_shot if entry.get("missed") or abs(entry["error_frames"]) > 2],
                                 "false_triggers": cross_false}
    ungated_rule = dict(best_rule, min_start_row=0)
    ungated_summary, _, ungated_false = combined(score_every_take(takes, grouped, [ungated_rule], chains_by_confidence)[0], names)
    output = {
        "date": time.strftime("%Y-%m-%d"), "machine": platform.node(), "split": "train only",
        "models": {"search": next(iter(takes.values()))["model"], "follow": next(iter(takes.values()))["follow_model"]},
        "coverage": {"takes": len(takes), "frames_processed": processed,
                     "processed_ranges": {name: take["ranges"] for name, take in takes.items()},
                     "how": f"each take from {MARGIN_AROUND_SHOTS} frames before its first training shot to {MARGIN_AROUND_SHOTS} after its last, "
                            f"every frame; test-shot windows (start_frame .. end_frame + {FRAMES_AFTER_LANDING_TO_EXCLUDE}) skipped and state reset"},
        "best_rule": best_rule, "best_summary": summary,
        "leave_one_take_out": leave_one_take_out_result,
        "best_rule_without_start_row_gate": {"rule": ungated_rule, "summary": ungated_summary,
                                             "false_triggers": [{key: trigger[key] for key in ("take", "fired_at", "where")} for trigger in ungated_false]},
        "top_rules": [{"rule": rules[k], "summary": combined(table[k], names)[0]} for _, k in scored[:15]],
        "idle_timing_mac_one_process": idle_timing({name: takes[name] for name in TAKES_TIMED_ALONE if name in takes}),
        "idle_timing_mac_two_processes_in_parallel": idle_timing({name: take for name, take in takes.items() if name not in TAKES_TIMED_ALONE}),
        "per_shot": per_shot, "false_triggers": false_triggers,
    }
    path = os.path.join(args.session, RESULT_NAME)
    write_json_atomically(path, output, indent=1)
    print(json.dumps({"best_rule": best_rule, "best_summary": summary, "leave_one_take_out": cross_summary, "idle_timing_mac": output["idle_timing_mac_one_process"]}, indent=1))
    print("saved", path)


def extrapolation_rules():
    for min_points in (3, 4, 5):
        for min_confidence in (0.25, 0.4):
            for min_start_row in (0, 420):
                for fit_points in (2, 3):
                    for start_row in (450, 480, 500, 520):
                        for offset in (-1, 0, 1):
                            yield {"min_points": min_points, "min_rise_px": 20, "min_detection_confidence": min_confidence,
                                   "max_span_frames": 20, "min_start_row": min_start_row, "estimate": "extrapolate-to-row",
                                   "fit_points": fit_points, "start_row": start_row, "offset": offset}


def frame_count_of(video_path):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_count


def cut_clip_by_stream_copy(video_path, first, last, clip_path):
    seek_seconds = (first + 0.25) / CAMERA_FPS
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{seek_seconds:.6f}", "-i", video_path, "-map", "0:v:0",
                    "-frames:v", str(last - first + 1), "-c", "copy", clip_path], check=True)


def write_frame_times(session, take, first, last, times_path):
    with open(os.path.join(session, take + ".csv"), newline="") as f:
        rows = [row for row in csv.DictReader(f) if first <= int(row["frame"]) <= last]
    with open(times_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_ranges(ranges, train_windows):
    gaps = []
    for first, last in ranges:
        cursor = first
        for window_first, window_last in sorted(train_windows):
            if window_last < cursor or window_first > last:
                continue
            if window_first > cursor:
                gaps.append([cursor, window_first - 1])
            cursor = max(cursor, window_last + 1)
        if cursor <= last:
            gaps.append([cursor, last])
    return gaps


def baseline_rule(session):
    with open(os.path.join(session, RESULT_NAME)) as f:
        return json.load(f)["best_rule"]


def make_package(args):
    grouped = shots_by_take(read_shots(args.session))
    os.makedirs(os.path.join(args.package, CLIP_FOLDER), exist_ok=True)
    takes = {}
    for take in args.takes or PACKAGE_TAKES:
        video_path = os.path.join(args.session, take + ".avi")
        frame_count = frame_count_of(video_path)
        ranges = processed_ranges(grouped[take], frame_count)
        clips = []
        for first, last in ranges:
            stem = f"{take}_{first:05d}_{last:05d}"
            clip_file = f"{CLIP_FOLDER}/{stem}.avi"
            times_file = f"{CLIP_FOLDER}/{stem}.csv"
            cut_clip_by_stream_copy(video_path, first, last, os.path.join(args.package, clip_file))
            write_frame_times(args.session, take, first, last, os.path.join(args.package, times_file))
            clips.append({"file": clip_file, "frame_times_file": times_file, "first_frame": first, "last_frame": last})
            print(f"{clip_file}: frames {first}..{last}", flush=True)
        train = [shot for shot in grouped[take] if shot["split"] == "train"]
        takes[take] = {"take": take, "frame_count": frame_count, "clips": clips, "train_shots": train,
                       "skipped_test_windows": [list(window) for window in excluded_test_ranges(grouped[take])],
                       "train_shot_windows": [list(shot_window(shot)) for shot in train],
                       "gap_frames": gap_ranges(ranges, [shot_window(shot) for shot in train])}
    manifest = {"made": time.strftime("%Y-%m-%d"), "source_session": args.session,
                "frame_numbers": "original take numbering; clip frame k is take frame first_frame + k",
                "rule": baseline_rule(args.session), "rule_source": os.path.join(args.session, RESULT_NAME),
                "takes": takes}
    write_json_atomically(os.path.join(args.package, MANIFEST_NAME), manifest, indent=1)
    print("saved", os.path.join(args.package, MANIFEST_NAME))


def frame_digest(frame):
    return hashlib.md5(frame.tobytes()).hexdigest()


def verify_package(args):
    manifest = read_manifest(args.package)
    for take, take_entry in sorted(manifest["takes"].items()):
        original = cv2.VideoCapture(os.path.join(manifest["source_session"], take + ".avi"))
        for clip in take_entry["clips"]:
            cut = cv2.VideoCapture(os.path.join(args.package, clip["file"]))
            expected_frames = clip["last_frame"] - clip["first_frame"] + 1
            original.set(cv2.CAP_PROP_POS_FRAMES, clip["first_frame"])
            compared, different = 0, []
            for frame_index in range(clip["first_frame"], clip["last_frame"] + 1):
                ok_original, original_frame = original.read()
                ok_cut, cut_frame = cut.read()
                if not (ok_original and ok_cut):
                    different.append(frame_index)
                    break
                compared += 1
                if frame_digest(original_frame) != frame_digest(cut_frame):
                    different.append(frame_index)
            extra_frame_in_cut, _ = cut.read()
            cut.release()
            print(f"{clip['file']}: {compared} of {expected_frames} frames compared, {len(different)} different, "
                  f"extra frames after the range: {extra_frame_in_cut}" + (f", first different {different[:5]}" if different else ""), flush=True)
        original.release()


def evaluate_package(args):
    manifest = read_manifest(args.package)
    rule = manifest["rule"]
    takes = load_cache(cache_folder(args))
    grouped = {take: manifest["takes"][take]["train_shots"] for take in takes}
    chains = {rule["min_detection_confidence"]: {name: chains_per_frame(take, rule["min_detection_confidence"]) for name, take in takes.items()}}
    names = sorted(takes)
    summary, per_shot, false_triggers = combined(score_every_take(takes, grouped, [rule], chains)[0], names)
    output = {"date": time.strftime("%Y-%m-%d"), "machine": platform.node(), "detector": args.detector, "overlap": args.overlap,
              "models": {"search": next(iter(takes.values()))["model"], "follow": next(iter(takes.values()))["follow_model"]},
              "rule": rule, "summary": summary, "waiting_timing": idle_timing(takes),
              "run_seconds": {name: take["run_seconds"] for name, take in takes.items()},
              "per_shot": per_shot, "false_triggers": false_triggers}
    path = os.path.join(args.package, f"hit_results_{args.detector}_{platform.node().split('.')[0]}{overlap_suffix(args)}.json")
    write_json_atomically(path, output, indent=1)
    print(json.dumps({"summary": summary, "waiting_timing": output["waiting_timing"]}, indent=1))
    for entry in per_shot:
        print(f"shot {entry['shot']} ({entry['take']}): label {entry['hit_label']}, "
              + ("missed" if entry.get("missed") else f"estimate {entry['hit_estimate']} ({entry['error_frames']:+d}), fired {entry['delay_frames']} frames after"))
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description="Hit detection from the rising chain over whole training takes, no hit label.")
    parser.add_argument("command", choices=("detect", "evaluate", "package", "verify-package"))
    parser.add_argument("--session", default="data/recordings/clears_nov10")
    parser.add_argument("--package")
    parser.add_argument("--detector", choices=sorted(MODELS_BY_DETECTOR), default="onnx")
    parser.add_argument("--model")
    parser.add_argument("--follow-model")
    parser.add_argument("--takes", nargs="+")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--overlap", action="store_true", help="run the next frame's blob search while the detector runs; own cache and results file")
    args = parser.parse_args()
    if args.command in ("package", "verify-package") and not args.package:
        parser.error(f"{args.command} needs --package")
    commands = {"detect": run_detection, "evaluate": run_evaluation, "package": make_package, "verify-package": verify_package}
    commands[args.command](args)


if __name__ == "__main__":
    main()
