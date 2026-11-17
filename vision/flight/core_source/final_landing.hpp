#include <vector>

constexpr double REFINE_INITIAL_DAMPING = 1e-3;
constexpr int REFINE_DAMPING_TRIES = 6;
constexpr double REFINE_STEP_TOLERANCE = 1e-9;

struct SmoothedStep {
    StateVector state;
    StateMatrix covariance;
};

static StateMatrix inverted_covariance(StateMatrix matrix) {
    std::array<bool, STATE_SIZE> fixed{};
    for (int i = 0; i < STATE_SIZE; ++i) {
        fixed[i] = matrix[i][i] <= 0.0;
        if (fixed[i]) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                matrix[i][j] = 0.0;
                matrix[j][i] = 0.0;
            }
            matrix[i][i] = 1.0;
        }
    }
    StateMatrix inverse = identity_state_matrix();
    for (int pivot = 0; pivot < STATE_SIZE; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < STATE_SIZE; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) {
                best = row;
            }
        }
        std::swap(matrix[pivot], matrix[best]);
        std::swap(inverse[pivot], inverse[best]);
        const double scale = 1.0 / matrix[pivot][pivot];
        for (int column = 0; column < STATE_SIZE; ++column) {
            matrix[pivot][column] *= scale;
            inverse[pivot][column] *= scale;
        }
        for (int row = 0; row < STATE_SIZE; ++row) {
            if (row == pivot || matrix[row][pivot] == 0.0) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (int column = 0; column < STATE_SIZE; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
                inverse[row][column] -= factor * inverse[pivot][column];
            }
        }
    }
    for (int i = 0; i < STATE_SIZE; ++i) {
        if (fixed[i]) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                inverse[i][j] = 0.0;
                inverse[j][i] = 0.0;
            }
        }
    }
    return inverse;
}

static std::vector<SmoothedStep> rts_smoothed(const KalmanFlight &filter) {
    const std::vector<FilterStep> &steps = filter.steps();
    std::vector<SmoothedStep> smoothed(steps.size());
    if (steps.empty()) {
        return smoothed;
    }
    smoothed.back() = {steps.back().state, steps.back().covariance};
    for (int index = static_cast<int>(steps.size()) - 2; index >= 0; --index) {
        const FilterStep &now = steps[index];
        const FilterStep &next = steps[index + 1];
        const StateMatrix gain = multiply(multiply_by_transpose(now.covariance, next.transition), inverted_covariance(next.predicted_covariance));
        SmoothedStep &result = smoothed[index];
        result.state = now.state;
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                result.state[i] += gain[i][j] * (smoothed[index + 1].state[j] - next.predicted_state[j]);
            }
        }
        StateMatrix covariance_change{};
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                covariance_change[i][j] = smoothed[index + 1].covariance[i][j] - next.predicted_covariance[i][j];
            }
        }
        const StateMatrix spread = sandwiched(gain, covariance_change);
        for (int i = 0; i < STATE_SIZE; ++i) {
            for (int j = 0; j < STATE_SIZE; ++j) {
                result.covariance[i][j] = now.covariance[i][j] + spread[i][j];
            }
        }
        result.covariance = symmetrized(result.covariance);
    }
    return smoothed;
}

struct RefineSettings {
    int max_iterations = 5;
    bool regate = false;
    double sigma_v_px = -1.0;
    double sigma_disparity_px = -1.0;
    double terminal_speed_sigma_m_s = -1.0;
};

static double setting_or_filter(double refine_value, double filter_value) {
    return refine_value < 0.0 ? filter_value : refine_value;
}

struct RefinedLanding {
    StateVector state;
    Vector landing;
    int iterations;
    int points_used;
    double cost_start;
    double cost_end;
};

struct TrackPoint {
    PixelMeasurement measurement;
    MeasurementVector sigmas;
    bool used;
};

class FinalLandingRefiner {
  public:
    FinalLandingRefiner(const KalmanFlight &filter, const RefineSettings &refine) : filter_(filter) {
        const KalmanSettings &settings = filter.settings();
        prior_sigma_ = setting_or_filter(refine.terminal_speed_sigma_m_s, settings.terminal_speed_sigma_m_s);
        sigma_v_px_ = setting_or_filter(refine.sigma_v_px, settings.sigma_v_px);
        sigma_disparity_px_ = setting_or_filter(refine.sigma_disparity_px, settings.sigma_disparity_px);
        parameter_count_ = prior_sigma_ > 0.0 ? STATE_SIZE : TERMINAL_SPEED_INDEX;
        for (const PixelMeasurement &measurement : filter.initial_measurements()) {
            points_.push_back({measurement, sigmas_of(measurement.values), true});
        }
        for (const FilterStep &step : filter.steps()) {
            points_.push_back({{step.time_s, step.measurement}, sigmas_of(step.measurement), step.accepted});
        }
        std::sort(points_.begin(), points_.end(), [](const TrackPoint &a, const TrackPoint &b) { return a.measurement.time_s > b.measurement.time_s; });
        reference_time_s_ = filter.steps().empty() ? 0.0 : filter.steps().back().time_s;
    }

    RefinedLanding refined(const RefineSettings &refine) {
        StateVector state = filter_.current_state();
        RefinedLanding result{state, filter_.landing_of(state), 0, used_count(), cost_of(state), 0.0};
        if (filter_.steps().empty() || used_count() < 3) {
            result.cost_end = result.cost_start;
            return result;
        }
        result.iterations = fitted(state, refine.max_iterations);
        if (refine.regate && regated(state)) {
            result.iterations += fitted(state, refine.max_iterations);
        }
        result.state = state;
        result.landing = filter_.landing_of(state);
        result.points_used = used_count();
        result.cost_end = cost_of(state);
        return result;
    }

  private:
    MeasurementVector sigmas_of(const MeasurementVector &values) const {
        const KalmanSettings &settings = filter_.settings();
        const double sigma_v = values[1] < settings.top_edge_v_px ? sigma_v_px_ * settings.top_edge_sigma_v_factor : sigma_v_px_;
        return {settings.sigma_u_px, sigma_v, sigma_disparity_px_};
    }

    int used_count() const {
        return static_cast<int>(std::count_if(points_.begin(), points_.end(), [](const TrackPoint &point) { return point.used; }));
    }

    StateVector propagated_by(const StateVector &state, double dt) const {
        const KalmanSettings &settings = filter_.settings();
        const double drag = filter_.drag_of(state);
        FlightState flight{{state[0], state[1], state[2]}, {state[3], state[4], state[5]}};
        const double direction = dt < 0.0 ? -1.0 : 1.0;
        double remaining = std::abs(dt);
        while (remaining > 1e-12) {
            const double step = std::min(settings.step_s, remaining);
            flight = rk4_step(flight, drag, settings.gravity, direction * step);
            remaining -= step;
        }
        return {flight.position.x, flight.position.y, flight.position.z, flight.velocity.x, flight.velocity.y, flight.velocity.z,
                state[TERMINAL_SPEED_INDEX]};
    }

    std::vector<MeasurementVector> whitened_residuals(const StateVector &state) const {
        std::vector<MeasurementVector> residuals(points_.size());
        StateVector along = state;
        double clock = reference_time_s_;
        for (std::size_t index = 0; index < points_.size(); ++index) {
            const TrackPoint &point = points_[index];
            along = propagated_by(along, point.measurement.time_s - clock);
            clock = point.measurement.time_s;
            const MeasurementVector expected = filter_.camera().measurement({along[0], along[1], along[2]});
            for (int i = 0; i < MEASUREMENT_SIZE; ++i) {
                residuals[index][i] = (point.measurement.values[i] - expected[i]) / point.sigmas[i];
            }
        }
        return residuals;
    }

    std::vector<double> residual_vector(const StateVector &state) const {
        const std::vector<MeasurementVector> residuals = whitened_residuals(state);
        std::vector<double> result;
        result.reserve(MEASUREMENT_SIZE * points_.size() + 1);
        for (std::size_t index = 0; index < points_.size(); ++index) {
            if (points_[index].used) {
                result.insert(result.end(), residuals[index].begin(), residuals[index].end());
            }
        }
        if (prior_sigma_ > 0.0) {
            result.push_back((state[TERMINAL_SPEED_INDEX] - filter_.settings().terminal_speed_m_s) / prior_sigma_);
        }
        return result;
    }

    double cost_of(const StateVector &state) const {
        double cost = 0.0;
        for (double value : residual_vector(state)) {
            cost += value * value;
        }
        return cost;
    }

    bool solved(StateMatrix normal, StateVector right_side, StateVector &solution) const {
        const int n = parameter_count_;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                double sum = normal[i][j];
                for (int k = 0; k < j; ++k) {
                    sum -= normal[i][k] * normal[j][k];
                }
                normal[i][j] = sum / normal[j][j];
            }
            double diagonal = normal[i][i];
            for (int k = 0; k < i; ++k) {
                diagonal -= normal[i][k] * normal[i][k];
            }
            if (diagonal <= 0.0) {
                return false;
            }
            normal[i][i] = std::sqrt(diagonal);
        }
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < i; ++k) {
                right_side[i] -= normal[i][k] * right_side[k];
            }
            right_side[i] /= normal[i][i];
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int k = i + 1; k < n; ++k) {
                right_side[i] -= normal[k][i] * right_side[k];
            }
            right_side[i] /= normal[i][i];
        }
        solution = StateVector{};
        for (int i = 0; i < n; ++i) {
            solution[i] = right_side[i];
        }
        return true;
    }

    int fitted(StateVector &state, int max_iterations) const {
        double damping = REFINE_INITIAL_DAMPING;
        std::vector<double> residuals = residual_vector(state);
        double cost = 0.0;
        for (double value : residuals) {
            cost += value * value;
        }
        int iterations = 0;
        for (; iterations < max_iterations; ++iterations) {
            std::vector<std::vector<double>> jacobian(parameter_count_);
            for (int column = 0; column < parameter_count_; ++column) {
                const double step = 1e-6 * std::max(1.0, std::abs(state[column]));
                StateVector ahead = state, behind = state;
                ahead[column] += step;
                behind[column] -= step;
                const std::vector<double> forward = residual_vector(ahead), backward = residual_vector(behind);
                jacobian[column].resize(residuals.size());
                for (std::size_t row = 0; row < residuals.size(); ++row) {
                    jacobian[column][row] = (forward[row] - backward[row]) / (2.0 * step);
                }
            }
            StateMatrix normal{};
            StateVector gradient{};
            for (int i = 0; i < parameter_count_; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = 0.0;
                    for (std::size_t row = 0; row < residuals.size(); ++row) {
                        sum += jacobian[i][row] * jacobian[j][row];
                    }
                    normal[i][j] = sum;
                    normal[j][i] = sum;
                }
                double sum = 0.0;
                for (std::size_t row = 0; row < residuals.size(); ++row) {
                    sum += jacobian[i][row] * residuals[row];
                }
                gradient[i] = -sum;
            }
            bool improved = false;
            StateVector step{};
            for (int attempt = 0; attempt < REFINE_DAMPING_TRIES && !improved; ++attempt) {
                StateMatrix damped = normal;
                for (int i = 0; i < parameter_count_; ++i) {
                    damped[i][i] *= 1.0 + damping;
                }
                if (!solved(damped, gradient, step)) {
                    damping *= 10.0;
                    continue;
                }
                StateVector candidate = state;
                for (int i = 0; i < parameter_count_; ++i) {
                    candidate[i] += step[i];
                }
                const std::vector<double> candidate_residuals = residual_vector(candidate);
                double candidate_cost = 0.0;
                for (double value : candidate_residuals) {
                    candidate_cost += value * value;
                }
                if (candidate_cost < cost) {
                    state = candidate;
                    residuals = candidate_residuals;
                    cost = candidate_cost;
                    damping = std::max(damping / 10.0, 1e-9);
                    improved = true;
                } else {
                    damping *= 10.0;
                }
            }
            if (!improved) {
                break;
            }
            double step_size = 0.0;
            for (int i = 0; i < parameter_count_; ++i) {
                step_size = std::max(step_size, std::abs(step[i]));
            }
            if (step_size < REFINE_STEP_TOLERANCE) {
                ++iterations;
                break;
            }
        }
        return iterations;
    }

    bool regated(const StateVector &state) {
        const std::vector<MeasurementVector> residuals = whitened_residuals(state);
        bool changed = false;
        for (std::size_t index = 0; index < points_.size(); ++index) {
            double squared = 0.0;
            for (double value : residuals[index]) {
                squared += value * value;
            }
            const bool inside = squared <= filter_.settings().gate_mahalanobis_sq;
            changed = changed || inside != points_[index].used;
            points_[index].used = inside;
        }
        return changed && used_count() >= 3;
    }

    const KalmanFlight &filter_;
    std::vector<TrackPoint> points_;
    double reference_time_s_ = 0.0;
    double prior_sigma_ = 0.0;
    double sigma_v_px_ = 0.0;
    double sigma_disparity_px_ = 0.0;
    int parameter_count_ = STATE_SIZE;
};

static RefinedLanding refine_final_landing(const KalmanFlight &filter, const RefineSettings &refine) {
    return FinalLandingRefiner(filter, refine).refined(refine);
}

static py::dict smoothed_history_dict(const KalmanFlight &filter) {
    const std::vector<SmoothedStep> smoothed = rts_smoothed(filter);
    const auto count = static_cast<py::ssize_t>(smoothed.size());
    py::array_t<double> states({count, static_cast<py::ssize_t>(STATE_SIZE)});
    py::array_t<double> covariances({count, static_cast<py::ssize_t>(STATE_SIZE), static_cast<py::ssize_t>(STATE_SIZE)});
    for (py::ssize_t index = 0; index < count; ++index) {
        for (int i = 0; i < STATE_SIZE; ++i) {
            states.mutable_data()[index * STATE_SIZE + i] = smoothed[index].state[i];
            for (int j = 0; j < STATE_SIZE; ++j) {
                covariances.mutable_data()[(index * STATE_SIZE + i) * STATE_SIZE + j] = smoothed[index].covariance[i][j];
            }
        }
    }
    py::dict result;
    result["states"] = states;
    result["covariances"] = covariances;
    return result;
}

static py::object smoothed_landing(const KalmanFlight &filter) {
    const std::vector<SmoothedStep> smoothed = rts_smoothed(filter);
    if (smoothed.empty()) {
        return py::none();
    }
    return array_from(filter.landing_of(smoothed.back().state));
}

static py::object refined_landing_dict(const KalmanFlight &filter, int max_iterations, bool regate, double sigma_v_px, double sigma_disparity_px,
                                       double terminal_speed_sigma_m_s) {
    if (!filter.initialized()) {
        return py::none();
    }
    const RefinedLanding refined = refine_final_landing(filter, {max_iterations, regate, sigma_v_px, sigma_disparity_px, terminal_speed_sigma_m_s});
    py::dict result;
    result["landing"] = array_from(refined.landing);
    py::array_t<double> state(STATE_SIZE);
    for (int i = 0; i < STATE_SIZE; ++i) {
        state.mutable_data()[i] = refined.state[i];
    }
    result["state"] = state;
    result["iterations"] = refined.iterations;
    result["points_used"] = refined.points_used;
    result["cost_start"] = refined.cost_start;
    result["cost_end"] = refined.cost_end;
    return result;
}

static void bind_final_landing(py::module_ &module) {
    module.def("smoothed_history", &smoothed_history_dict, py::arg("filter"));
    module.def("smoothed_landing", &smoothed_landing, py::arg("filter"));
    module.def("refine_final_landing", &refined_landing_dict, py::arg("filter"), py::arg("max_iterations") = 5, py::arg("regate") = false,
               py::arg("sigma_v_px") = -1.0, py::arg("sigma_disparity_px") = -1.0, py::arg("terminal_speed_sigma_m_s") = -1.0);
}
