import multiprocessing
import queue
import time

from flight_model import corrected_landing_cm, corrected_landing_with_warm_start, drag_per_metre

STOP = None


def fit_in_background(requests, results, drag, correction):
    warm_start, warm_shot = None, None
    while True:
        request = requests.get()
        if request is STOP:
            return
        shot, frame, times, points_m = request
        if shot != warm_shot:
            warm_start, warm_shot = None, shot
        started = time.perf_counter()
        landing_cm, warm_start = corrected_landing_with_warm_start(times, points_m, drag, correction, warm_start=warm_start)
        results.put((shot, frame, landing_cm, (time.perf_counter() - started) * 1000))


class LandingPredictor:
    def __init__(self, terminal_speed_m_s, correction, background=True):
        self.drag = drag_per_metre(terminal_speed_m_s)
        self.correction = correction
        self.background = background
        self.shot = 0
        self.busy = False
        self.inline_result = None
        if background:
            context = multiprocessing.get_context("spawn")
            self.requests, self.results = context.Queue(), context.Queue()
            self.worker = context.Process(target=fit_in_background, args=(self.requests, self.results, self.drag, correction), daemon=True)
            self.worker.start()

    def start_shot(self):
        self.drain_until_idle()
        self.shot += 1
        self.busy = False
        self.inline_result = None

    def submit(self, frame, times, points_m):
        if self.busy:
            return False
        if self.background:
            self.requests.put((self.shot, frame, times, points_m))
            self.busy = True
        else:
            started = time.perf_counter()
            landing_cm = corrected_landing_cm(times, points_m, self.drag, self.correction)
            self.inline_result = (frame, landing_cm, (time.perf_counter() - started) * 1000)
        return True

    def poll(self):
        if not self.background:
            result, self.inline_result = self.inline_result, None
            return result
        newest = None
        while True:
            try:
                shot, frame, landing_cm, fit_ms = self.results.get_nowait()
            except queue.Empty:
                return newest
            if shot == self.shot:
                self.busy = False
                newest = (frame, landing_cm, fit_ms)

    def drain_until_idle(self):
        while self.background and self.busy:
            shot, frame, landing_cm, fit_ms = self.results.get()
            if shot == self.shot:
                self.busy = False

    def final_landing_cm(self, times, points_m):
        self.drain_until_idle()
        return corrected_landing_cm(times, points_m, self.drag, self.correction)

    def close(self):
        if self.background:
            self.requests.put(STOP)
            self.worker.join(timeout=5)
