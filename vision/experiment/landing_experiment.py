#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court_3d import StereoCourt
from flight_model import (CM_PER_M, DEFAULT_INTEGRATION, Integration, acceleration, apply_correction, drag_per_metre,
                          fit_flight, landing_point, observed_flight)
from frames import load_frame_times

RK2_COMPARISON_STEP_S = 0.002
TERMINAL_SPEEDS_TO_TRY_M_S = np.arange(4.5, 10.01, 0.25)


def rk2_step(position, velocity, drag, dt):
    a1 = acceleration(velocity, drag)
    v_mid = velocity + 0.5 * dt * a1
    a2 = acceleration(v_mid, drag)
    return position + dt * v_mid, velocity + dt * a2


INTEGRATIONS = {
    "rk4": DEFAULT_INTEGRATION,
    "rk2": Integration("rk2", rk2_step, RK2_COMPARISON_STEP_S),
}


def load_shots(session_dir, tracks_name="bird_tracks.json"):
    with open(os.path.join(session_dir, "shots.csv"), newline="") as f:
        shots = {int(row["shot"]): row for row in csv.DictReader(f)}
    with open(os.path.join(session_dir, tracks_name)) as f:
        tracks = {track["shot"]: track for track in json.load(f)}
    return shots, tracks


def predicted_landings_cm(flights, drag, integration=DEFAULT_INTEGRATION):
    predictions = {}
    for shot_number, (times, points_m) in flights.items():
        position, velocity = fit_flight(times, points_m, drag, integration)
        predictions[shot_number] = landing_point(position, velocity, drag, integration)[:2] * CM_PER_M
    return predictions


def fit_correction(predicted, actual):
    design = np.column_stack([predicted, np.ones(len(predicted))])
    coefficients, *_ = np.linalg.lstsq(design, actual, rcond=None)
    return coefficients


def error_report(name, predicted, actual):
    miss = predicted - actual
    distance = np.linalg.norm(miss, axis=1)
    return {"method": name, "mean_miss_cm": float(distance.mean()), "median_miss_cm": float(np.median(distance)),
            "mean_abs_x_cm": float(np.abs(miss[:, 0]).mean()), "mean_abs_y_cm": float(np.abs(miss[:, 1]).mean())}


def run(session_dir, landing_x_sign, tracks_name="bird_tracks.json", integration=DEFAULT_INTEGRATION):
    started = time.perf_counter()
    court = StereoCourt.from_session(session_dir)
    shots, tracks = load_shots(session_dir, tracks_name)
    take_times = {take: load_frame_times(os.path.join(session_dir, take + ".avi"))[0] for take in {s["take"] for s in shots.values()}}
    flights, skipped = {}, []
    for number, shot in shots.items():
        flight = observed_flight(session_dir, court, shot, tracks[number], take_times)
        if flight is None:
            skipped.append(number)
        else:
            flights[number] = flight
    actual = {n: np.array([landing_x_sign * float(s["landing_x"]), float(s["landing_y"])]) for n, s in shots.items()}
    train = [n for n in flights if shots[n]["split"] == "train"]
    test = [n for n in flights if shots[n]["split"] == "test"]

    best = None
    for terminal_speed in TERMINAL_SPEEDS_TO_TRY_M_S:
        drag = drag_per_metre(terminal_speed)
        predictions = predicted_landings_cm({n: flights[n] for n in train}, drag, integration)
        miss = np.mean([np.linalg.norm(predictions[n] - actual[n]) for n in train])
        if best is None or miss < best[0]:
            best = (miss, terminal_speed)
    terminal_speed = best[1]
    drag = drag_per_metre(terminal_speed)
    predictions = predicted_landings_cm(flights, drag, integration)

    train_predicted = np.array([predictions[n] for n in train])
    train_actual = np.array([actual[n] for n in train])
    test_predicted = np.array([predictions[n] for n in test])
    test_actual = np.array([actual[n] for n in test])
    correction = fit_correction(train_predicted, train_actual)
    train_mean = train_actual.mean(axis=0)

    reports = [
        error_report("guess the training average", np.tile(train_mean, (len(test), 1)), test_actual),
        error_report("physics only", test_predicted, test_actual),
        error_report("physics + correction learned on train", apply_correction(correction, test_predicted), test_actual),
    ]
    per_shot = [{"shot": n, "actual_cm": actual[n].round(0).tolist(), "physics_cm": predictions[n].round(0).tolist(),
                 "corrected_cm": apply_correction(correction, predictions[n][None, :])[0].round(0).tolist()} for n in test]
    return {"tracks": tracks_name, "landing_x_sign": landing_x_sign, "terminal_speed_m_s": float(terminal_speed), "skipped_shots": skipped,
            "train_count": len(train), "test_count": len(test), "test_results": reports, "test_shots": per_shot,
            "correction": correction.round(3).tolist(),
            "integrator": integration.name, "integration_step_ms": integration.step_s * 1000,
            "run_seconds": round(time.perf_counter() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description="Predict test-shot landings from motion tracks, tuned on the training shots.")
    parser.add_argument("session")
    parser.add_argument("--landing-x-sign", type=float, default=-1.0,
                        help="multiply recorded landing_x by this (-1 = recorded on the left-post side without a sign)")
    parser.add_argument("--tracks", default="bird_tracks.json", help="tracks file in the session folder")
    parser.add_argument("--output-name", default="landing_experiment.json")
    parser.add_argument("--integrator", choices=sorted(INTEGRATIONS), default=DEFAULT_INTEGRATION.name,
                        help="rk4 = 4 acceleration calls per step, used by the pipeline (default); rk2 = midpoint, 2 calls per step, kept for comparison")
    parser.add_argument("--step-ms", type=float, help="integration step; defaults to 25 ms for rk4 and 2 ms for rk2")
    args = parser.parse_args()
    integration = INTEGRATIONS[args.integrator]
    if args.step_ms:
        integration = integration._replace(step_s=args.step_ms / 1000)
    result = run(args.session, args.landing_x_sign, args.tracks, integration)
    output = os.path.join(args.session, args.output_name)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"integrator {result['integrator']} every {result['integration_step_ms']:g} ms, {result['run_seconds']} s")
    print(f"tuned shuttle terminal speed: {result['terminal_speed_m_s']:.2f} m/s  | train {result['train_count']}  test {result['test_count']}  skipped {result['skipped_shots']}")
    for report in result["test_results"]:
        print(f"{report['method']:<40} mean miss {report['mean_miss_cm']:5.0f} cm  median {report['median_miss_cm']:5.0f}  |x| {report['mean_abs_x_cm']:4.0f}  |y| {report['mean_abs_y_cm']:4.0f}")
    print("saved", output)


if __name__ == "__main__":
    main()
