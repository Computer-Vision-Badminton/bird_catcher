#include <cmath>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Vector {
    double x, y, z;
};

static Vector operator+(Vector a, Vector b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vector operator*(double s, Vector a) { return {s * a.x, s * a.y, s * a.z}; }
static Vector operator/(Vector a, double s) { return {a.x / s, a.y / s, a.z / s}; }
static Vector operator-(Vector a, Vector b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

struct FlightState {
    Vector position;
    Vector velocity;
};

static Vector acceleration(Vector velocity, double drag, double gravity) {
    const double speed = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    const double brake = drag * speed;
    return {0.0 - brake * velocity.x, 0.0 - brake * velocity.y, -gravity - brake * velocity.z};
}

static FlightState rk4_step(FlightState state, double drag, double gravity, double dt) {
    const Vector k1_velocity = state.velocity;
    const Vector k1_acceleration = acceleration(k1_velocity, drag, gravity);
    const Vector k2_velocity = state.velocity + (0.5 * dt) * k1_acceleration;
    const Vector k2_acceleration = acceleration(k2_velocity, drag, gravity);
    const Vector k3_velocity = state.velocity + (0.5 * dt) * k2_acceleration;
    const Vector k3_acceleration = acceleration(k3_velocity, drag, gravity);
    const Vector k4_velocity = state.velocity + dt * k3_acceleration;
    const Vector k4_acceleration = acceleration(k4_velocity, drag, gravity);
    const Vector average_velocity = (k1_velocity + 2.0 * k2_velocity + 2.0 * k3_velocity + k4_velocity) / 6.0;
    const Vector average_acceleration = (k1_acceleration + 2.0 * k2_acceleration + 2.0 * k3_acceleration + k4_acceleration) / 6.0;
    return {state.position + dt * average_velocity, state.velocity + dt * average_acceleration};
}

static Vector vector_from(py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    if (values.size() != 3) {
        throw std::invalid_argument("expected 3 values (x, y, z)");
    }
    const double *data = values.data();
    return {data[0], data[1], data[2]};
}

static py::array_t<double> positions_at(py::array_t<double, py::array::c_style | py::array::forcecast> start_position,
                                        py::array_t<double, py::array::c_style | py::array::forcecast> start_velocity,
                                        double drag, py::array_t<double, py::array::c_style | py::array::forcecast> times,
                                        double step_s, double gravity) {
    FlightState state{vector_from(start_position), vector_from(start_velocity)};
    const auto count = times.size();
    const double *targets = times.data();
    py::array_t<double> positions({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(3)});
    double *out = positions.mutable_data();
    double clock = 0.0;
    for (py::ssize_t index = 0; index < count; ++index) {
        const double target = targets[index];
        while (clock + step_s <= target) {
            state = rk4_step(state, drag, gravity, step_s);
            clock += step_s;
        }
        const double partial = target - clock;
        const Vector sampled = partial > 1e-9 ? rk4_step(state, drag, gravity, partial).position : state.position;
        out[3 * index] = sampled.x;
        out[3 * index + 1] = sampled.y;
        out[3 * index + 2] = sampled.z;
    }
    return positions;
}

static py::array_t<double> landing_point(py::array_t<double, py::array::c_style | py::array::forcecast> start_position,
                                         py::array_t<double, py::array::c_style | py::array::forcecast> start_velocity,
                                         double drag, double step_s, double max_flight_s, double gravity) {
    FlightState state{vector_from(start_position), vector_from(start_velocity)};
    double clock = 0.0;
    Vector landing = state.position;
    while (clock < max_flight_s) {
        const FlightState next = rk4_step(state, drag, gravity, step_s);
        if (next.position.z <= 0.0) {
            const double fraction = state.position.z / (state.position.z - next.position.z);
            landing = state.position + fraction * (next.position - state.position);
            break;
        }
        state = next;
        clock += step_s;
        landing = state.position;
    }
    py::array_t<double> result(3);
    double *out = result.mutable_data();
    out[0] = landing.x;
    out[1] = landing.y;
    out[2] = landing.z;
    return result;
}

PYBIND11_MODULE(flight_core, module) {
    module.def("positions_at", &positions_at, py::arg("start_position"), py::arg("start_velocity"), py::arg("drag"),
               py::arg("times"), py::arg("step_s"), py::arg("gravity"));
    module.def("landing_point", &landing_point, py::arg("start_position"), py::arg("start_velocity"), py::arg("drag"),
               py::arg("step_s"), py::arg("max_flight_s"), py::arg("gravity"));
}
