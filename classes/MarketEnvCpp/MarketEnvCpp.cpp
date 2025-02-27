#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

float clip(float value, float lower, float upper) {
    return std::max(lower, std::min(value, upper));
}

float calculate_std(const std::vector<float>& data) {
    if (data.empty()) return 0.0f;
    
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float accum = 0.0f;
    for (float value : data) {
        accum += (value - mean) * (value - mean);
    }
    
    return std::sqrt(accum / data.size());
}

float normalize_reward(float reward) {
    if (std::abs(reward) > 1.0f) {
        if (reward < 0) {
            return -std::log(1 + std::abs(reward));
        } else {
            return std::log(1 + std::abs(reward));
        }
    }
    return 0.0f;
}

std::tuple<float, float, float> calculate_pnl_metrics(float current_price, 
                                                      float entry_price, 
                                                      float trading_fee, 
                                                      float portfolio_value, 
                                                      py::array_t<float> returns_window) {

    py::buffer_info buf = returns_window.request();
    float* pointer = static_cast<float*>(buf.ptr);
    std::vector<float> returns_vector(pointer, pointer + buf.size);

    float pnl_pct = ((current_price - entry_price) / (std::abs(entry_price)) + 0.00001f);
    float trade_cost_pct = (trading_fee * 2) / portfolio_value;
    float net_pnl_pct = pnl_pct - trade_cost_pct;
    float risk_adjusted_pnl = 0.0f;

    // Compute volatility and risk-adjusted PnL if the returns window has more than one element
    if (returns_vector.size() > 1) {
        float volatility = calculate_std(returns_vector);
        volatility = clip(volatility, 0.001f, 0.5f);
        risk_adjusted_pnl = net_pnl_pct / (volatility + 1e-4f);
    } else {
        risk_adjusted_pnl = net_pnl_pct;
    }

    return std::make_tuple(trade_cost_pct, net_pnl_pct, risk_adjusted_pnl);
}

float calculate_local_sharpe(
    py::array_t<float> returns,
    int lookback
    ) {
    float risk_free_rate = 0.02f;
    py::buffer_info buf = returns.request();
    float* pointer = static_cast<float*>(buf.ptr);
    
    int total_size = buf.size;
    if (total_size < lookback) {
        return 0.0f;
    }
    
    float* start_ptr = pointer + (total_size - lookback);
    
    float sum = 0.0f;
    for (int i = 0; i < lookback; i++) {
        sum += start_ptr[i];
    }
    float avg_return = sum / lookback;
    
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < lookback; i++) {
        float diff = start_ptr[i] - avg_return;
        sum_sq_diff += diff * diff;
    }
    float std_return = std::sqrt(sum_sq_diff / lookback);
    
    if (std_return == 0.0f) {
        return 0.0f;
    }
    
    return (avg_return - risk_free_rate) / std_return;
}

PYBIND11_MODULE(MarketEnvCpp, m) {
    m.doc() = "Fast C++ implementations for MarketEnv";
    m.def("normalize_reward", &normalize_reward, "Calculate normalized trading reward");
    m.def("calculate_pnl_metrics", &calculate_pnl_metrics, "Calculate PnL metrics");
    m.def("calculate_local_sharpe", &calculate_local_sharpe, "Calculate local sharpe");
}
