#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <pybind11/stl.h>

constexpr int STATE_SIZE = 7;
constexpr int MEASUREMENT_SIZE = 3;
constexpr int TERMINAL_SPEED_INDEX = 6;
constexpr double MIN_TERMINAL_SPEED_M_S = 2.0;
constexpr int UNDISTORT_ITERATIONS = 20;

using StateVector = std::array<double, STATE_SIZE>;
using StateMatrix = std::array<std::array<double, STATE_SIZE>, STATE_SIZE>;
using MeasurementVector = std::array<double, MEASUREMENT_SIZE>;
using MeasurementMatrix = std::array<std::array<double, MEASUREMENT_SIZE>, MEASUREMENT_SIZE>;
using MeasurementJacobian = std::array<std::array<double, STATE_SIZE>, MEASUREMENT_SIZE>;
using GainMatrix = std::array<std::array<double, MEASUREMENT_SIZE>, STATE_SIZE>;

static StateMatrix zero_state_matrix() {
    StateMatrix result{};
    return result;
}

static StateMatrix identity_state_matrix() {
    StateMatrix result{};
    for (int i = 0; i < STATE_SIZE; ++i) {
        result[i][i] = 1.0;
    }
    return result;
}

static StateMatrix multiply(const StateMatrix &a, const StateMatrix &b) {
    StateMatrix result{};
    for (int i = 0; i < STATE_SIZE; ++i) {
        for (int k = 0; k < STATE_SIZE; ++k) {
            if (a[i][k] == 0.0) {
                continue;
            }
            for (int j = 0; j < STATE_SIZE; ++j) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

static StateMatrix multiply_by_transpose(const StateMatrix &a, const StateMatrix &b) {
    StateMatrix result{};
    for (int i = 0; i < STATE_SIZE; ++i) {
        for (int j = 0; j < STATE_SIZE; ++j) {
            double sum = 0.0;
            for (int k = 0; k < STATE_SIZE; ++k) {
                sum += a[i][k] * b[j][k];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

static StateMatrix sandwiched(const StateMatrix &outer, const StateMatrix &inner) {
    return multiply_by_transpose(multiply(outer, inner), outer);
}

static StateMatrix symmetrized(const StateMatrix &matrix) {
    StateMatrix result = matrix;
    for (int i = 0; i < STATE_SIZE; ++i) {
        for (int j = i + 1; j < STATE_SIZE; ++j) {
            const double average = 0.5 * (matrix[i][j] + matrix[j][i]);
            result[i][j] = average;
            result[j][i] = average;
        }
    }
    return result;
}

static MeasurementMatrix inverted(const MeasurementMatrix &m) {
    const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double determinant = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    const double scale = 1.0 / determinant;
    MeasurementMatrix result{};
    result[0][0] = c00 * scale;
    result[1][0] = c01 * scale;
    result[2][0] = c02 * scale;
    result[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * scale;
    result[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * scale;
    result[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * scale;
    result[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * scale;
    result[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * scale;
    result[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * scale;
    return result;
}

struct ImagePoint {
    double column, row;
};

struct StereoCamera {
    double focal_px;
    double principal_column_px;
    double principal_row_px;
    double right_row_offset_px;
    double k1;
    double k2;
    double baseline_m;
    std::array<double, 9> camera_from_court;
    Vector translation_m;

    Vector camera_point(Vector court_m) const {
        const auto &m = camera_from_court;
        return {m[0] * court_m.x + m[1] * court_m.y + m[2] * court_m.z + translation_m.x,
                m[3] * court_m.x + m[4] * court_m.y + m[5] * court_m.z + translation_m.y,
                m[6] * court_m.x + m[7] * court_m.y + m[8] * court_m.z + translation_m.z};
    }

    Vector court_point_from_camera(Vector camera_m) const {
        const auto &m = camera_from_court;
        const Vector shifted = camera_m - translation_m;
        return {m[0] * shifted.x + m[3] * shifted.y + m[6] * shifted.z, m[1] * shifted.x + m[4] * shifted.y + m[7] * shifted.z,
                m[2] * shifted.x + m[5] * shifted.y + m[8] * shifted.z};
    }

    double distortion_factor(double x, double y) const {
        const double radius_sq = x * x + y * y;
        return 1.0 + k1 * radius_sq + k2 * radius_sq * radius_sq;
    }

    ImagePoint pixel(Vector camera_m, double principal_row) const {
        const double x = camera_m.x / camera_m.z;
        const double y = camera_m.y / camera_m.z;
        const double factor = distortion_factor(x, y);
        return {focal_px * x * factor + principal_column_px, focal_px * y * factor + principal_row};
    }

    ImagePoint normalized(ImagePoint pixel_point, double principal_row) const {
        const double distorted_x = (pixel_point.column - principal_column_px) / focal_px;
        const double distorted_y = (pixel_point.row - principal_row) / focal_px;
        double x = distorted_x, y = distorted_y;
        for (int iteration = 0; iteration < UNDISTORT_ITERATIONS; ++iteration) {
            const double factor = distortion_factor(x, y);
            x = distorted_x / factor;
            y = distorted_y / factor;
        }
        return {x, y};
    }

    MeasurementVector measurement(Vector court_m) const {
        const Vector left_camera = camera_point(court_m);
        const Vector right_camera = left_camera - Vector{baseline_m, 0.0, 0.0};
        const ImagePoint left = pixel(left_camera, principal_row_px);
        const ImagePoint right = pixel(right_camera, principal_row_px + right_row_offset_px);
        return {left.column, left.row, left.column - right.column};
    }

    Vector court_point(double left_column, double left_row, double disparity) const {
        const double right_principal_row = principal_row_px + right_row_offset_px;
        const ImagePoint left = normalized({left_column, left_row}, principal_row_px);
        const ImagePoint right = normalized({left_column - disparity, left_row + right_row_offset_px}, right_principal_row);
        const double depth = baseline_m / (left.column - right.column);
        return court_point_from_camera({left.column * depth, left.row * depth, depth});
    }
};

struct KalmanSettings {
    double sigma_u_px = 0.5;
    double sigma_v_px = 2.0;
    double sigma_disparity_px = 0.2;
    double top_edge_v_px = 12.0;
    double top_edge_sigma_v_factor = 5.0;
    double gate_mahalanobis_sq = 16.3;
    double acceleration_noise_m2_s3 = 0.0;
    double terminal_speed_m_s = 6.5;
    double terminal_speed_sigma_m_s = 0.0;
    double terminal_speed_walk_m2_s3 = 0.0;
    int initial_points = 3;
    double initial_sigma_scale = 3.0;
    double initial_velocity_sigma_m_s = 1.0;
    int reset_after_rejections = 3;
    double step_s = 0.025;
    double max_flight_s = 4.0;
    double gravity = 9.81;
};

struct PixelMeasurement {
    double time_s;
    MeasurementVector values;
};

struct FilterStep {
    double time_s;
    MeasurementVector measurement;
    StateVector predicted_state;
    StateMatrix predicted_covariance;
    StateVector state;
    StateMatrix covariance;
    StateMatrix transition;
    bool accepted;
    double mahalanobis_sq;
};

class KalmanFlight {
  public:
    KalmanFlight(StereoCamera camera, KalmanSettings settings) : camera_(camera), settings_(settings) {}

    py::object update(double time_s, double u, double v, double disparity) {
        const PixelMeasurement measurement{time_s, {u, v, disparity}};
        if (!initialized_) {
            pending_.push_back(measurement);
            if (static_cast<int>(pending_.size()) < std::max(settings_.initial_points, 2)) {
                return py::none();
            }
            initialize_from(pending_);
            pending_.clear();
        } else if (!filter_step(measurement, true)) {
            pending_.push_back(measurement);
            if (static_cast<int>(pending_.size()) >= std::max(settings_.reset_after_rejections, 2)) {
                initialize_from(pending_);
                ++reset_count_;
                pending_.clear();
            }
        } else {
            pending_.clear();
        }
        return array_from(landing());
    }

    Vector landing() const {
        return landing_of(state_);
    }

    bool initialized() const { return initialized_; }

    int rejected_count() const { return rejected_count_; }

    int reset_count() const { return reset_count_; }

    py::array_t<double> state_array() const { return vector_array(state_); }

    py::array_t<double> covariance_array() const { return matrix_array(covariance_); }

    py::array_t<double> landing_covariance() const {
        const double terminal_speed_step = 1e-4;
        std::array<std::array<double, STATE_SIZE>, 2> jacobian{};
        for (int column = 0; column < STATE_SIZE; ++column) {
            const double step = column == TERMINAL_SPEED_INDEX ? terminal_speed_step : 1e-5;
            StateVector ahead = state_, behind = state_;
            ahead[column] += step;
            behind[column] -= step;
            const Vector forward = landing_of(ahead), backward = landing_of(behind);
            jacobian[0][column] = (forward.x - backward.x) / (2.0 * step);
            jacobian[1][column] = (forward.y - backward.y) / (2.0 * step);
        }
        py::array_t<double> result({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
        double *out = result.mutable_data();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                double sum = 0.0;
                for (int a = 0; a < STATE_SIZE; ++a) {
                    for (int b = 0; b < STATE_SIZE; ++b) {
                        sum += jacobian[i][a] * covariance_[a][b] * jacobian[j][b];
                    }
                }
                out[2 * i + j] = sum;
            }
        }
        return result;
    }

    py::dict history() const {
        const auto count = static_cast<py::ssize_t>(steps_.size());
        py::array_t<double> times(count), mahalanobis(count);
        py::array_t<bool> accepted(count);
        py::array_t<double> measurements({count, static_cast<py::ssize_t>(MEASUREMENT_SIZE)});
        py::array_t<double> predicted_states({count, static_cast<py::ssize_t>(STATE_SIZE)});
        py::array_t<double> states({count, static_cast<py::ssize_t>(STATE_SIZE)});
        const std::vector<py::ssize_t> matrix_shape{count, STATE_SIZE, STATE_SIZE};
        py::array_t<double> predicted_covariances(matrix_shape), covariances(matrix_shape), transitions(matrix_shape);
        for (py::ssize_t index = 0; index < count; ++index) {
            const FilterStep &step = steps_[index];
            times.mutable_data()[index] = step.time_s;
            mahalanobis.mutable_data()[index] = step.mahalanobis_sq;
            accepted.mutable_data()[index] = step.accepted;
            for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
                measurements.mutable_data()[index * MEASUREMENT_SIZE + i] = step.measurement[i];
            }
            for (int i = 0; i < STATE_SIZE; ++i) {
                predicted_states.mutable_data()[index * STATE_SIZE + i] = step.predicted_state[i];
                states.mutable_data()[index * STATE_SIZE + i] = step.state[i];
                for (int j = 0; j < STATE_SIZE; ++j) {
                    const py::ssize_t offset = (index * STATE_SIZE + i) * STATE_SIZE + j;
                    predicted_covariances.mutable_data()[offset] = step.predicted_covariance[i][j];
                    covariances.mutable_data()[offset] = step.covariance[i][j];
                    transitions.mutable_data()[offset] = step.transition[i][j];
                }
            }
        }
        py::dict result;
        result["times_s"] = times;
        result["measurements"] = measurements;
        result["predicted_states"] = predicted_states;
        result["predicted_covariances"] = predicted_covariances;
        result["states"] = states;
        result["covariances"] = covariances;
        result["transitions"] = transitions;
        result["accepted"] = accepted;
        result["mahalanobis_sq"] = mahalanobis;
        return result;
    }

    StateVector propagated(const StateVector &state, double dt) const {
        const double drag = drag_of(state);
        FlightState flight{{state[0], state[1], state[2]}, {state[3], state[4], state[5]}};
        double remaining = dt;
        while (remaining > 1e-12) {
            const double step = std::min(settings_.step_s, remaining);
            flight = rk4_step(flight, drag, settings_.gravity, step);
            remaining -= step;
        }
        return {flight.position.x, flight.position.y, flight.position.z, flight.velocity.x, flight.velocity.y, flight.velocity.z,
                state[TERMINAL_SPEED_INDEX]};
    }

    double drag_of(const StateVector &state) const {
        const double terminal_speed = std::max(state[TERMINAL_SPEED_INDEX], MIN_TERMINAL_SPEED_M_S);
        return settings_.gravity / (terminal_speed * terminal_speed);
    }

    Vector landing_of(const StateVector &state) const {
        const FlightState flight{{state[0], state[1], state[2]}, {state[3], state[4], state[5]}};
        return landing_from(flight, drag_of(state), settings_.step_s, settings_.max_flight_s, settings_.gravity);
    }

    MeasurementVector measurement_sigmas(const MeasurementVector &values) const {
        const double sigma_v = values[1] < settings_.top_edge_v_px ? settings_.sigma_v_px * settings_.top_edge_sigma_v_factor : settings_.sigma_v_px;
        return {settings_.sigma_u_px, sigma_v, settings_.sigma_disparity_px};
    }

    const StereoCamera &camera() const { return camera_; }

    const KalmanSettings &settings() const { return settings_; }

    const StateVector &current_state() const { return state_; }

    const std::vector<FilterStep> &steps() const { return steps_; }

    const std::vector<PixelMeasurement> &initial_measurements() const { return initial_measurements_; }

  private:

    StateMatrix transition_jacobian(const StateVector &state, double dt) const {
        StateMatrix jacobian{};
        for (int column = 0; column < STATE_SIZE; ++column) {
            const double step = 1e-6 * std::max(1.0, std::abs(state[column]));
            StateVector ahead = state, behind = state;
            ahead[column] += step;
            behind[column] -= step;
            const StateVector forward = propagated(ahead, dt), backward = propagated(behind, dt);
            for (int row = 0; row < STATE_SIZE; ++row) {
                jacobian[row][column] = (forward[row] - backward[row]) / (2.0 * step);
            }
        }
        return jacobian;
    }

    StateMatrix process_noise(double dt) const {
        StateMatrix noise{};
        const double q = settings_.acceleration_noise_m2_s3;
        for (int axis = 0; axis < 3; ++axis) {
            noise[axis][axis] = q * dt * dt * dt / 3.0;
            noise[axis][axis + 3] = q * dt * dt / 2.0;
            noise[axis + 3][axis] = q * dt * dt / 2.0;
            noise[axis + 3][axis + 3] = q * dt;
        }
        if (settings_.terminal_speed_sigma_m_s > 0.0) {
            noise[TERMINAL_SPEED_INDEX][TERMINAL_SPEED_INDEX] = settings_.terminal_speed_walk_m2_s3 * dt;
        }
        return noise;
    }

    MeasurementJacobian measurement_jacobian(const StateVector &state) const {
        MeasurementJacobian jacobian{};
        const double step = 1e-6;
        for (int column = 0; column < 3; ++column) {
            StateVector ahead = state, behind = state;
            ahead[column] += step;
            behind[column] -= step;
            const MeasurementVector forward = camera_.measurement({ahead[0], ahead[1], ahead[2]});
            const MeasurementVector backward = camera_.measurement({behind[0], behind[1], behind[2]});
            for (int row = 0; row < MEASUREMENT_SIZE; ++row) {
                jacobian[row][column] = (forward[row] - backward[row]) / (2.0 * step);
            }
        }
        return jacobian;
    }

    StateMatrix triangulation_covariance(const PixelMeasurement &measurement) const {
        const MeasurementVector sigmas = measurement_sigmas(measurement.values);
        std::array<Vector, MEASUREMENT_SIZE> columns{};
        for (int input = 0; input < MEASUREMENT_SIZE; ++input) {
            const double step = 1e-3;
            MeasurementVector ahead = measurement.values, behind = measurement.values;
            ahead[input] += step;
            behind[input] -= step;
            const Vector forward = camera_.court_point(ahead[0], ahead[1], ahead[2]);
            const Vector backward = camera_.court_point(behind[0], behind[1], behind[2]);
            columns[input] = (sigmas[input] / (2.0 * step)) * (forward - backward);
        }
        StateMatrix covariance{};
        for (const Vector &column : columns) {
            const std::array<double, 3> values{column.x, column.y, column.z};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    covariance[i][j] += values[i] * values[j];
                }
            }
        }
        return covariance;
    }

    void initialize_from(const std::vector<PixelMeasurement> &measurements) {
        const PixelMeasurement &first = measurements.front();
        const PixelMeasurement &last = measurements.back();
        const Vector first_point = camera_.court_point(first.values[0], first.values[1], first.values[2]);
        const Vector last_point = camera_.court_point(last.values[0], last.values[1], last.values[2]);
        const double span = std::max(last.time_s - first.time_s, 1e-3);
        const Vector mean_velocity = (last_point - first_point) / span;
        const double drag = settings_.gravity / (settings_.terminal_speed_m_s * settings_.terminal_speed_m_s);
        const Vector end_velocity = mean_velocity + (0.5 * span) * acceleration(mean_velocity, drag, settings_.gravity);
        state_ = {last_point.x, last_point.y, last_point.z, end_velocity.x, end_velocity.y, end_velocity.z, settings_.terminal_speed_m_s};
        const StateMatrix first_covariance = triangulation_covariance(first);
        const StateMatrix last_covariance = triangulation_covariance(last);
        const double scale_sq = settings_.initial_sigma_scale * settings_.initial_sigma_scale;
        const double velocity_floor = settings_.initial_velocity_sigma_m_s * settings_.initial_velocity_sigma_m_s;
        covariance_ = zero_state_matrix();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covariance_[i][j] = scale_sq * last_covariance[i][j];
                covariance_[i][j + 3] = scale_sq * last_covariance[i][j] / span;
                covariance_[i + 3][j] = scale_sq * last_covariance[i][j] / span;
                covariance_[i + 3][j + 3] = scale_sq * (first_covariance[i][j] + last_covariance[i][j]) / (span * span);
            }
            covariance_[i + 3][i + 3] += velocity_floor;
        }
        covariance_[TERMINAL_SPEED_INDEX][TERMINAL_SPEED_INDEX] = settings_.terminal_speed_sigma_m_s * settings_.terminal_speed_sigma_m_s;
        last_time_s_ = last.time_s;
        initialized_ = true;
        initial_measurements_.assign(measurements.begin(), measurements.end() - 1);
        steps_.clear();
        FilterStep step{};
        step.time_s = last.time_s;
        step.measurement = last.values;
        step.transition = identity_state_matrix();
        step.predicted_state = state_;
        step.predicted_covariance = covariance_;
        step.state = state_;
        step.covariance = covariance_;
        step.accepted = true;
        steps_.push_back(step);
    }

    bool filter_step(const PixelMeasurement &measurement, bool gate) {
        const double dt = measurement.time_s - last_time_s_;
        FilterStep step{};
        step.time_s = measurement.time_s;
        step.measurement = measurement.values;
        if (dt > 0.0) {
            step.transition = transition_jacobian(state_, dt);
            state_ = propagated(state_, dt);
            StateMatrix predicted = sandwiched(step.transition, covariance_);
            const StateMatrix noise = process_noise(dt);
            for (int i = 0; i < STATE_SIZE; ++i) {
                for (int j = 0; j < STATE_SIZE; ++j) {
                    predicted[i][j] += noise[i][j];
                }
            }
            covariance_ = symmetrized(predicted);
            last_time_s_ = measurement.time_s;
        } else {
            step.transition = identity_state_matrix();
        }
        step.predicted_state = state_;
        step.predicted_covariance = covariance_;
        step.accepted = correct(measurement.values, gate, step.mahalanobis_sq);
        if (!step.accepted) {
            ++rejected_count_;
        }
        step.state = state_;
        step.covariance = covariance_;
        steps_.push_back(step);
        return step.accepted;
    }

    bool correct(const MeasurementVector &values, bool gate, double &mahalanobis_sq) {
        const MeasurementVector expected = camera_.measurement({state_[0], state_[1], state_[2]});
        const MeasurementJacobian h = measurement_jacobian(state_);
        const MeasurementVector sigmas = measurement_sigmas(values);
        MeasurementVector innovation{};
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            innovation[i] = values[i] - expected[i];
        }
        std::array<std::array<double, MEASUREMENT_SIZE>, STATE_SIZE> covariance_times_ht{};
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += covariance_[i][k] * h[j][k];
                }
                covariance_times_ht[i][j] = sum;
            }
        }
        MeasurementMatrix innovation_covariance{};
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 3; ++k) {
                    sum += h[i][k] * covariance_times_ht[k][j];
                }
                innovation_covariance[i][j] = sum;
            }
            innovation_covariance[i][i] += sigmas[i] * sigmas[i];
        }
        const MeasurementMatrix inverse = inverted(innovation_covariance);
        mahalanobis_sq = 0.0;
        for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
            for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                mahalanobis_sq += innovation[i] * inverse[i][j] * innovation[j];
            }
        }
        if (gate && mahalanobis_sq > settings_.gate_mahalanobis_sq) {
            return false;
        }
        GainMatrix gain{};
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < MEASUREMENT_SIZE; ++k) {
                    sum += covariance_times_ht[i][k] * inverse[k][j];
                }
                gain[i][j] = sum;
            }
        }
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                state_[i] += gain[i][j] * innovation[j];
            }
        }
        StateMatrix keep = identity_state_matrix();
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < MEASUREMENT_SIZE; ++j) {
                    keep[i][k] -= gain[i][j] * h[j][k];
                }
            }
        }
        StateMatrix updated = sandwiched(keep, covariance_);
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < MEASUREMENT_SIZE; ++k) {
                    sum += gain[i][k] * sigmas[k] * sigmas[k] * gain[j][k];
                }
                updated[i][j] += sum;
            }
        }
        covariance_ = symmetrized(updated);
        return true;
    }

    static py::array_t<double> vector_array(const StateVector &values) {
        py::array_t<double> result(STATE_SIZE);
        for (int i = 0; i < STATE_SIZE; ++i) {
            result.mutable_data()[i] = values[i];
        }
        return result;
    }

    static py::array_t<double> matrix_array(const StateMatrix &values) {
        py::array_t<double> result({static_cast<py::ssize_t>(STATE_SIZE), static_cast<py::ssize_t>(STATE_SIZE)});
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                result.mutable_data()[i * STATE_SIZE + j] = values[i][j];
            }
        }
        return result;
    }

    StereoCamera camera_;
    KalmanSettings settings_;
    StateVector state_{};
    StateMatrix covariance_{};
    bool initialized_ = false;
    double last_time_s_ = 0.0;
    int rejected_count_ = 0;
    int reset_count_ = 0;
    std::vector<PixelMeasurement> pending_;
    std::vector<FilterStep> steps_;
    std::vector<PixelMeasurement> initial_measurements_;
};

static StereoCamera stereo_camera_from(double focal_px, std::array<double, 2> principal_point_px, double right_row_offset_px,
                                       std::array<double, 2> distortion_k1_k2, double baseline_m,
                                       std::array<std::array<double, 3>, 3> camera_from_court, std::array<double, 3> translation_m) {
    StereoCamera camera{};
    camera.focal_px = focal_px;
    camera.principal_column_px = principal_point_px[0];
    camera.principal_row_px = principal_point_px[1];
    camera.right_row_offset_px = right_row_offset_px;
    camera.k1 = distortion_k1_k2[0];
    camera.k2 = distortion_k1_k2[1];
    camera.baseline_m = baseline_m;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            camera.camera_from_court[3 * row + column] = camera_from_court[row][column];
        }
    }
    camera.translation_m = {translation_m[0], translation_m[1], translation_m[2]};
    return camera;
}

static void bind_kalman_flight(py::module_ &module) {
    py::class_<StereoCamera>(module, "StereoCamera")
        .def(py::init(&stereo_camera_from), py::arg("focal_px"), py::arg("principal_point_px"), py::arg("right_row_offset_px"),
             py::arg("distortion_k1_k2"), py::arg("baseline_m"), py::arg("camera_from_court"), py::arg("translation_m"))
        .def("measurement", [](const StereoCamera &camera, std::array<double, 3> court_m) {
            const MeasurementVector values = camera.measurement({court_m[0], court_m[1], court_m[2]});
            return array_from({values[0], values[1], values[2]});
        }, py::arg("court_m"))
        .def("court_point", [](const StereoCamera &camera, double u, double v, double disparity) {
            return array_from(camera.court_point(u, v, disparity));
        }, py::arg("u"), py::arg("v"), py::arg("disparity"));

    py::class_<KalmanSettings>(module, "KalmanSettings")
        .def(py::init<>())
        .def_readwrite("sigma_u_px", &KalmanSettings::sigma_u_px)
        .def_readwrite("sigma_v_px", &KalmanSettings::sigma_v_px)
        .def_readwrite("sigma_disparity_px", &KalmanSettings::sigma_disparity_px)
        .def_readwrite("top_edge_v_px", &KalmanSettings::top_edge_v_px)
        .def_readwrite("top_edge_sigma_v_factor", &KalmanSettings::top_edge_sigma_v_factor)
        .def_readwrite("gate_mahalanobis_sq", &KalmanSettings::gate_mahalanobis_sq)
        .def_readwrite("acceleration_noise_m2_s3", &KalmanSettings::acceleration_noise_m2_s3)
        .def_readwrite("terminal_speed_m_s", &KalmanSettings::terminal_speed_m_s)
        .def_readwrite("terminal_speed_sigma_m_s", &KalmanSettings::terminal_speed_sigma_m_s)
        .def_readwrite("terminal_speed_walk_m2_s3", &KalmanSettings::terminal_speed_walk_m2_s3)
        .def_readwrite("initial_points", &KalmanSettings::initial_points)
        .def_readwrite("initial_sigma_scale", &KalmanSettings::initial_sigma_scale)
        .def_readwrite("reset_after_rejections", &KalmanSettings::reset_after_rejections)
        .def_readwrite("initial_velocity_sigma_m_s", &KalmanSettings::initial_velocity_sigma_m_s)
        .def_readwrite("step_s", &KalmanSettings::step_s)
        .def_readwrite("max_flight_s", &KalmanSettings::max_flight_s)
        .def_readwrite("gravity", &KalmanSettings::gravity);

    py::class_<KalmanFlight>(module, "KalmanFlight")
        .def(py::init<StereoCamera, KalmanSettings>(), py::arg("camera"), py::arg("settings"))
        .def("update", &KalmanFlight::update, py::arg("time_s"), py::arg("u"), py::arg("v"), py::arg("disparity"))
        .def("landing", [](const KalmanFlight &filter) { return array_from(filter.landing()); })
        .def("landing_covariance", &KalmanFlight::landing_covariance)
        .def("state", &KalmanFlight::state_array)
        .def("covariance", &KalmanFlight::covariance_array)
        .def("history", &KalmanFlight::history)
        .def_property_readonly("initialized", &KalmanFlight::initialized)
        .def_property_readonly("rejected_count", &KalmanFlight::rejected_count)
        .def_property_readonly("reset_count", &KalmanFlight::reset_count);
}
