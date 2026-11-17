import collections
import time

from vision.tracking.bird_tracker import FRAMES_AFTER_HIT
from vision.detection.crop_tracking import CROP_SIZE, FOLLOW_CROP_SIZE
from vision.detection.detectors import GRAY_STACK_INPUT, STEREO_BATCH, load_detector
from vision.tracking.hit_detection import HitWatcher, WaitingDetector, milliseconds_since
from vision.pipeline.shot_pipeline import LANDING_BY_KALMAN, SEARCH_WITH_FOLLOW_ONCE_SEEN, ShotPipeline

RECENT_FRAMES_KEPT = 24
SEARCHING = "searching"
FOLLOWING_BEFORE_HIT = "following"
TRACKING = "tracking"
STATES = (SEARCHING, FOLLOWING_BEFORE_HIT, TRACKING)
WAITING_BATCH = 1
HAILO_SEARCH_AT_BATCH_1 = "search-batch-1"
HAILO_SHARED_BATCH_2 = "shared-batch-2"
HAILO_MODEL_CHOICES = (HAILO_SEARCH_AT_BATCH_1, HAILO_SHARED_BATCH_2)
SEEDED_FROM_FOLLOWING = "seeded"
REPLAY_FALLBACK = "replay_fallback"
REPLAY_ALWAYS = "replay"


def load_live_detectors(detector_kind, search_model, follow_model, hailo_models=HAILO_SEARCH_AT_BATCH_1):
    def load(model, batch, device_of):
        return load_detector(detector_kind, model, input_kind=GRAY_STACK_INPUT, hailo_batch_size=batch, on_hailo_device_of=device_of)

    if detector_kind != "hailo":
        search, follow = load(search_model, STEREO_BATCH, None), load(follow_model, STEREO_BATCH, None)
        return {"waiting_search": search, "waiting_follow": follow, "tracking_search": search, "tracking_follow": follow}
    tracking_search = load(search_model, STEREO_BATCH, None)
    tracking_follow = load(follow_model, STEREO_BATCH, tracking_search)
    if hailo_models == HAILO_SHARED_BATCH_2:
        return {"waiting_search": tracking_search, "waiting_follow": tracking_follow,
                "tracking_search": tracking_search, "tracking_follow": tracking_follow}
    return {"waiting_search": load(search_model, WAITING_BATCH, tracking_search), "waiting_follow": tracking_follow,
            "tracking_search": tracking_search, "tracking_follow": tracking_follow}


def distinct_detectors(detectors):
    return list({id(detector): detector for detector in detectors.values()}.values())


def crop_size_of(detectors, detector):
    return FOLLOW_CROP_SIZE if detector in (detectors["waiting_follow"], detectors["tracking_follow"]) else CROP_SIZE


def state_of_waiting_mode(mode):
    return SEARCHING if mode in (None, "search") else FOLLOWING_BEFORE_HIT


def added_stage_times(total, stages_ms):
    for stage, ms in stages_ms.items():
        total[stage] = total.get(stage, 0.0) + ms
    return total


def plain_landing(landing_cm):
    return None if landing_cm is None else [round(float(value), 2) for value in landing_cm]


class LiveShotLoop:
    def __init__(self, detectors, court, right_row_offset_px, terminal_speed_m_s, correction, overlap=True, forced_hits=None,
                 seed_tracking_from_following=True):
        self.detectors = detectors
        self.court = court
        self.right_row_offset_px = right_row_offset_px
        self.terminal_speed_m_s = terminal_speed_m_s
        self.correction = correction
        self.overlap = overlap
        self.forced_hits = forced_hits
        self.seed_tracking_from_following = seed_tracking_from_following
        self.waiting = WaitingDetector(detectors["waiting_search"], detectors["waiting_follow"], overlap,
                                       right_row_offset_px if seed_tracking_from_following else None)
        self.hit_watcher = HitWatcher()
        self.recent_frames = collections.deque(maxlen=RECENT_FRAMES_KEPT)
        self.stereo_detections = {}
        self.tracking = None
        self.hit_waiting_for_its_frame_time = None
        self.events = []
        self.shots = []
        self.replay_fallbacks = []

    def push(self, frame_index, stereo_frame, time_s):
        started = time.perf_counter()
        self.recent_frames.append((frame_index, stereo_frame, time_s))
        if self.tracking is None:
            record = self.wait_for_hit(frame_index, stereo_frame, time_s)
        else:
            record = self.track(frame_index, stereo_frame, time_s)
        record.update({"frame": frame_index, "t_s": time_s, "main_ms": round(milliseconds_since(started), 3)})
        self.note_hit_frame_time(record["main_ms"])
        return record

    def note_hit_frame_time(self, main_ms):
        if self.hit_waiting_for_its_frame_time is None:
            return
        for hit in self.hit_waiting_for_its_frame_time:
            hit["hit_frame_main_ms"] = main_ms
        self.hit_waiting_for_its_frame_time = None

    def remember_stereo_detections(self, detected_frame, entry):
        if "stereo_detections" not in entry:
            return
        self.stereo_detections[detected_frame] = {"detections": entry["stereo_detections"], "crop_size": entry["crop"][2]}
        for old_frame in [old_frame for old_frame in self.stereo_detections if old_frame <= detected_frame - RECENT_FRAMES_KEPT]:
            del self.stereo_detections[old_frame]

    def wait_for_hit(self, frame_index, stereo_frame, time_s):
        gray_ms, _, finished = self.waiting.step(stereo_frame, frame_index)
        record = {"state": state_of_waiting_mode(None if finished is None else finished[1]["mode"]),
                  "stages_ms": {"gray": round(gray_ms, 3)}}
        hit = None
        if finished is not None:
            detected_frame, entry = finished
            record["stages_ms"].update({"aim": entry["aim_ms"], "yolo": entry["yolo_ms"]})
            record.update({"detected_frame": detected_frame, "mode": entry["mode"], "detector_call_ms": entry["detector_call_ms"],
                           "detections": entry["detections"]})
            if "right_detections" in entry:
                record.update({"right_crop": entry["right_crop"], "right_detections": entry["right_detections"]})
            self.remember_stereo_detections(detected_frame, entry)
            if self.forced_hits is None:
                hit = self.hit_watcher.push(detected_frame, entry["detections"])
        if self.forced_hits is not None and frame_index in self.forced_hits:
            hit = {"fired_at": frame_index, "hit_estimate": self.forced_hits[frame_index], "forced": True}
        if hit is not None:
            self.start_tracking(hit, frame_index, time_s, record)
        return record

    def start_tracking(self, hit, frame_index, time_s, record):
        hit = dict(hit, fired_on_frame=frame_index, fired_on_time_s=time_s)
        estimate = hit["hit_estimate"]
        pipeline = ShotPipeline(self.detectors["tracking_search"], self.court, self.right_row_offset_px, self.terminal_speed_m_s,
                                self.correction, estimate, estimate + FRAMES_AFTER_HIT,
                                follow_detector=self.detectors["tracking_follow"],
                                search_with_follow_detector=SEARCH_WITH_FOLLOW_ONCE_SEEN,
                                overlap_detection=self.overlap, landing_method=LANDING_BY_KALMAN)
        first_needed = pipeline.first_frame_needed()
        kept_frames = [kept for kept in self.recent_frames if kept[0] >= first_needed]
        hit["replay_first_frame"] = kept_frames[0][0]
        hit["frames_missing_from_buffer"] = max(0, kept_frames[0][0] - first_needed)
        frames_detected_while_waiting = [frame for frame in pipeline.frames_to_track if frame < frame_index]
        running_frame = self.waiting.pending_stereo_frame()
        frames_without_stereo = [frame for frame in frames_detected_while_waiting
                                 if frame not in self.stereo_detections and frame != running_frame]
        seeded = self.seed_tracking_from_following and not frames_without_stereo
        running = self.waiting.hand_over_pending() if seeded and running_frame is not None else None
        drained = self.waiting.drain()
        if drained is not None:
            self.remember_stereo_detections(*drained)
            record["drained_detected_frame"] = drained[0]
        hit.update({"start": SEEDED_FROM_FOLLOWING if seeded else REPLAY_FALLBACK if self.seed_tracking_from_following else REPLAY_ALWAYS,
                    "seeded_frames": len(frames_detected_while_waiting) if seeded else 0,
                    "handed_over_running_frame": None if running is None else running[0],
                    "replayed_frames": 0 if seeded else len(kept_frames)})
        if self.seed_tracking_from_following and not seeded:
            hit["frames_without_stereo_detections"] = frames_without_stereo
            self.replay_fallbacks.append({"hit_estimate": estimate, "fired_on_frame": frame_index, "frames_without_stereo_detections": frames_without_stereo})
        self.tracking = {"pipeline": pipeline, "hit": hit, "estimates": [], "frames_tracked": 0, "frames_seeded": hit["seeded_frames"]}
        hit_event = {"event": "hit", **hit}
        self.events.append(hit_event)
        self.hit_waiting_for_its_frame_time = (hit, hit_event)
        record["event"] = "hit"
        stages_ms, detector_call_ms = {}, []
        if seeded:
            self.seed_tracking(pipeline, kept_frames, [frame for frame in frames_detected_while_waiting if frame != running_frame],
                               frame_index, stages_ms)
            if running is not None:
                pipeline.take_running_detection(*running)
        else:
            for kept_index, kept_frame, kept_time in kept_frames:
                self.push_to_pipeline(kept_index, kept_frame, kept_time, frame_index, stages_ms, detector_call_ms)
        record["tracking_stages_ms"] = {stage: round(ms, 3) for stage, ms in stages_ms.items()}
        record["tracking_detector_call_ms"] = detector_call_ms
        record.update({"start": hit["start"], "seeded_frames": hit["seeded_frames"], "replayed_frames": hit["replayed_frames"]})
        self.end_tracking_if_done(frame_index, record)

    def seed_tracking(self, pipeline, kept_frames, frames_to_seed, arrived_frame, stages_ms):
        for kept_index, _, kept_time in kept_frames:
            pipeline.take_known_frame_time(kept_index, kept_time)
        pipeline.take_known_grays({"left": list(self.waiting.grays), "right": self.waiting.right_eye_grays()})
        for frame in frames_to_seed:
            buffered = self.stereo_detections[frame]
            self.remember_result(pipeline.push_detected(frame, *buffered["detections"], buffered["crop_size"]), arrived_frame, stages_ms, [])

    def push_to_pipeline(self, frame_index, stereo_frame, time_s, arrived_frame, stages_ms, detector_call_ms):
        self.tracking["frames_tracked"] += 1
        self.remember_result(self.tracking["pipeline"].push(frame_index, stereo_frame, time_s), arrived_frame, stages_ms, detector_call_ms)

    def remember_result(self, result, arrived_frame, stages_ms, detector_call_ms):
        if result is None:
            return
        added_stage_times(stages_ms, result["stages_ms"])
        if result["detector_call_ms"] is not None:
            detector_call_ms.append(round(float(result["detector_call_ms"]), 3))
        pipeline = self.tracking["pipeline"]
        self.tracking["estimates"].append({"frame": result["frame"], "t_s": pipeline.frame_times[result["frame"]],
                                           "arrived_frame": arrived_frame, "crop_size": result["crop_size"],
                                           "landing_cm": plain_landing(result["landing_cm"]),
                                           "landing_from_frame": result["landing_from_frame"]})

    def track(self, frame_index, stereo_frame, time_s):
        record = {"state": TRACKING}
        stages_ms, detector_call_ms = {}, []
        self.push_to_pipeline(frame_index, stereo_frame, time_s, frame_index, stages_ms, detector_call_ms)
        record["stages_ms"] = {stage: round(ms, 3) for stage, ms in stages_ms.items()}
        record["detector_call_ms"] = detector_call_ms[0] if detector_call_ms else None
        self.end_tracking_if_done(frame_index, record)
        return record

    def end_tracking_if_done(self, frame_index, record):
        if frame_index >= self.tracking["pipeline"].last_frame_needed():
            self.end_tracking(frame_index, record)

    def end_tracking(self, frame_index, record, complete=True):
        pipeline = self.tracking["pipeline"]
        finish_stages, finish_calls = {}, []
        self.remember_result(pipeline.drain(), frame_index, finish_stages, finish_calls)
        started = time.perf_counter()
        final_landing = pipeline.finish()
        finish_stages["finish"] = milliseconds_since(started)
        record.setdefault("stages_ms", {})
        added_stage_times(record["stages_ms"], {stage: round(ms, 3) for stage, ms in finish_stages.items()})
        if finish_calls:
            record["drained_detector_call_ms"] = finish_calls[0]
        shot = {"hit": self.tracking["hit"], "complete": complete, "final_landing_cm": plain_landing(final_landing),
                "final_on_frame": frame_index, "final_on_time_s": pipeline.frame_times.get(frame_index),
                "frames_tracked": self.tracking["frames_tracked"], "frames_seeded": self.tracking["frames_seeded"], "stereo_points": len(pipeline.stereo),
                "estimates": self.tracking["estimates"]}
        self.shots.append(shot)
        self.events.append({"event": "landing", "hit_estimate": shot["hit"]["hit_estimate"], "final_landing_cm": shot["final_landing_cm"],
                            "final_on_frame": frame_index, "complete": complete})
        record["event"] = "landing"
        self.tracking = None
        self.stereo_detections = {}
        self.waiting.reset()
        self.hit_watcher.reset()

    def close(self):
        record = {"stages_ms": {}}
        if self.tracking is not None:
            self.end_tracking(self.recent_frames[-1][0], record, complete=False)
        else:
            self.waiting.drain()
        return record
