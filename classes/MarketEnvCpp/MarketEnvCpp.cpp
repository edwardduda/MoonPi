#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace py = pybind11;


float calculate_local_sharpe(
    const std::vector<float>& returns_vector,
    int& window_lookback
    ) {
    float risk_free_rate = 0.02f;
    
    if (returns_vector.size() < static_cast<size_t>(window_lookback)) {
        return 0.0f;
    }
    
    float sum = std::accumulate(returns_vector.begin(), returns_vector.end(), 0.0f);
    
    float avg_return = sum / window_lookback;
    
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < window_lookback; i++) {
        float diff = returns_vector[i] - avg_return;
        sum_sq_diff += diff * diff;
    }
    float std_return = std::sqrt(sum_sq_diff / window_lookback);
    
    if (std_return == 0.0f) {
        return 0.0f;
    }
    
    return (avg_return - risk_free_rate) / std_return;
}

float calculate_relative_strength(const std::vector<float>& prices, int current_idx, int window_lookback){
    if (current_idx < window_lookback) {
        return 0.0f;
    }

    float current_price = prices[current_idx];
    int start_idx = std::max(0, current_idx - window_lookback);

    float sum = std::accumulate(prices.begin() + start_idx, prices.begin() + current_idx, 0.0f);
    float avg_price = sum / (current_idx - start_idx);
    
    return (current_price - avg_price) / avg_price;
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
                                                      const std::vector<float>& returns_vector) {

    float pnl_pct = ((current_price - entry_price) / (std::abs(entry_price)) + 1e-5f);
    float trade_cost_pct = (trading_fee * 2) / portfolio_value;
    float net_pnl_pct = pnl_pct - trade_cost_pct;
    float risk_adjusted_pnl = 0.0f;

    // Compute volatility and risk-adjusted PnL if the returns window has more than one element
    if (returns_vector.size() > 1) {
        float volatility = calculate_std(returns_vector);
        volatility = std::max(0.001f, std::min(volatility, 0.5f));
        risk_adjusted_pnl = net_pnl_pct / (volatility + 1e-5f);
    } else {
        risk_adjusted_pnl = net_pnl_pct;
    }

    return std::make_tuple(trade_cost_pct, net_pnl_pct, risk_adjusted_pnl);
}

std::tuple<float, float, float> calculate_risk_metrics(float current_price,
                                                    int window_position,
                                                    int window_lookback,
                                                    const std::vector<float>& prices_vector,
                                                    const std::vector<float>& returns_vector){

    float sharpe = 0.0f;
    float relative_strength = 0.0f;
    float volatility = 0.0f;

    if(returns_vector.size() >= static_cast<size_t>(window_lookback)){
        sharpe = calculate_local_sharpe(returns_vector, window_lookback);
    }
    if(returns_vector.size() >= static_cast<size_t>(window_lookback)){
        volatility = calculate_std(returns_vector);
        volatility = std::max(0.001f, std::min(volatility, 0.5f));
    }
    if(returns_vector.size() >= static_cast<size_t>(window_lookback)){
        relative_strength = calculate_relative_strength(prices_vector, window_position, window_lookback);
    }
    
    return std::make_tuple(sharpe, relative_strength, volatility);
    }

PYBIND11_MODULE(MarketEnvCpp, m) {
    m.doc() = "Fast C++ implementations for MarketEnv";
    m.def("normalize_reward", &normalize_reward, "Calculate normalized trading reward");
    m.def("calculate_pnl_metrics", &calculate_pnl_metrics, "Calculate PnL metrics");
    m.def("calculate_local_sharpe", &calculate_local_sharpe, "Calculate local sharpe");
    m.def("calculate_relative_strength", &calculate_relative_strength, "Calculate relative strength");
    m.def("calculate_risk_metrics", &calculate_risk_metrics, "Calculate_risk_metrics");
}
