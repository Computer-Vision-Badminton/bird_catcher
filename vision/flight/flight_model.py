import collections

import numpy as np
from scipy.optimize import least_squares

try:
    from vision.flight import flight_core
except ImportError:
    flight_core = None

GRAVITY_M_S2 = 9.81
CM_PER_M = 100.0
FIRST_USABLE_FRAME_AFTER_HIT = 2
MIN_POINTS_FOR_FIT = 6
OUTLIER_RESIDUAL_M = 0.35
INTEGRATION_STEP_S = 0.025
MAX_FLIGHT_S = 4.0


def drag_per_metre(terminal_speed):
    return GRAVITY_M_S2 / terminal_speed ** 2


def acceleration(velocity, drag):
    return np.array([0.0, 0.0, -GRAVITY_M_S2]) - drag * np.linalg.norm(velocity) * velocity


def rk4_step(position, velocity, drag, dt):
    k1_velocity = velocity
    k1_acceleration = acceleration(k1_velocity, drag)
    k2_velocity = velocity + 0.5 * dt * k1_acceleration
    k2_acceleration = acceleration(k2_velocity, drag)
    k3_velocity = velocity + 0.5 * dt * k2_acceleration
    k3_acceleration = acceleration(k3_velocity, drag)
    k4_velocity = velocity + dt * k3_acceleration
    k4_acceleration = acceleration(k4_velocity, drag)
    average_velocity = (k1_velocity + 2 * k2_velocity + 2 * k3_velocity + k4_velocity) / 6
    average_acceleration = (k1_acceleration + 2 * k2_acceleration + 2 * k3_acceleration + k4_acceleration) / 6
    return position + dt * average_velocity, velocity + dt * average_acceleration


Integration = collections.namedtuple("Integration", "name step step_s")
DEFAULT_INTEGRATION = Integration("rk4", rk4_step, INTEGRATION_STEP_S)
FLIGHT_ENGINE = "c++" if flight_core is not None else "python"


def uses_compiled_flight(integration):
    return flight_core is not None and integration.step is rk4_step


def positions_at(start_position, start_velocity, drag, times, integration=DEFAULT_INTEGRATION):
    if uses_compiled_flight(integration):
        return flight_core.positions_at(start_position, start_velocity, drag, times, integration.step_s, GRAVITY_M_S2)
    position, velocity = np.array(start_position, float), np.array(start_velocity, float)
    clock = 0.0
    out = []
    for target in times:
        while clock + integration.step_s <= target:
            position, velocity = integration.step(position, velocity, drag, integration.step_s)
            clock += integration.step_s
        partial = target - clock
        if partial > 1e-9:
            out.append(integration.step(position, velocity, drag, partial)[0])
        else:
            out.append(position.copy())
    return np.array(out)


def landing_point(start_position, start_velocity, drag, integration=DEFAULT_INTEGRATION):
    if uses_compiled_flight(integration):
        return flight_core.landing_point(start_position, start_velocity, drag, integration.step_s, MAX_FLIGHT_S, GRAVITY_M_S2)
    position, velocity = np.array(start_position, float), np.array(start_velocity, float)
    clock = 0.0
    while clock < MAX_FLIGHT_S:
        next_position, next_velocity = integration.step(position, velocity, drag, integration.step_s)
        if next_position[2] <= 0:
            fraction = position[2] / (position[2] - next_position[2])
            return position + fraction * (next_position - position)
        position, velocity, clock = next_position, next_velocity, clock + integration.step_s
    return position


def fit_flight_with_warm_start(times, points_m, drag, integration=DEFAULT_INTEGRATION, warm_start=None):
    times = times - times[0]
    if warm_start is None:
        span = max(times[-1], 1e-3)
        warm_start = np.concatenate([points_m[0], (points_m[-1] - points_m[0]) / span])

    def residuals(params):
        return (positions_at(params[:3], params[3:], drag, times, integration) - points_m).ravel()

    first_fit = least_squares(residuals, np.asarray(warm_start, float), loss="soft_l1", f_scale=0.1)
    fit = first_fit
    errors = np.linalg.norm(residuals(fit.x).reshape(-1, 3), axis=1)
    keep = errors < OUTLIER_RESIDUAL_M
    if keep.sum() >= MIN_POINTS_FOR_FIT and not keep.all():
        times, points_m = times[keep], points_m[keep]
        fit = least_squares(lambda p: (positions_at(p[:3], p[3:], drag, times - times[0], integration) - points_m).ravel(), fit.x)
    return fit.x[:3], fit.x[3:], first_fit.x


def fit_flight(times, points_m, drag, integration=DEFAULT_INTEGRATION):
    position, velocity, _ = fit_flight_with_warm_start(times, points_m, drag, integration)
    return position, velocity


def flight_points(court, stereo_matches, hit_frame, frame_times):
    stereo = np.array([p for p in stereo_matches if p[0] - hit_frame >= FIRST_USABLE_FRAME_AFTER_HIT])
    if len(stereo) < MIN_POINTS_FOR_FIT:
        return None
    times = np.array([frame_times[int(frame)] for frame in stereo[:, 0]])
    points_m = court.court_points_cm(stereo[:, 1:3], stereo[:, 3:5]) / CM_PER_M
    return times, points_m


def observed_flight(session_dir, court, shot, track, take_times):
    return flight_points(court, track["stereo"], track["hit_frame"], take_times[shot["take"]])


def corrected_landing_with_warm_start(times, points_m, drag, correction, integration=DEFAULT_INTEGRATION, warm_start=None):
    position, velocity, next_warm_start = fit_flight_with_warm_start(times, points_m, drag, integration, warm_start)
    physics_cm = landing_point(position, velocity, drag, integration)[:2] * CM_PER_M
    return apply_correction(np.asarray(correction), physics_cm[None, :])[0], next_warm_start


def corrected_landing_cm(times, points_m, drag, correction, integration=DEFAULT_INTEGRATION):
    return corrected_landing_with_warm_start(times, points_m, drag, correction, integration)[0]


def apply_correction(coefficients, predicted):
    return np.column_stack([predicted, np.ones(len(predicted))]) @ coefficients
