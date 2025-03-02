#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <rapidcsv.h>
#include <DataFrame/DataFrame.h>
#include <DataFrame/Utils/DateTime.h>

namespace py = pybind11;


class MarketEnv {
    
private:

    float initial_capital;
    int max_trades_per_month; 
    float trading_fee;
    float hold_penalty;
    int max_hold_steps;
    int segment_size;
    int num_projected_days;

    int lookback = 30;
    std::string full_data;
        

    std::vector<float> sharpe_window;
    std::vector<float> volatility_window;
    int risk_feature_dim = 3;
    //Get feature dimensions (excluding Close and Ticker)
    //self.feature_columns = [col for col in self.full_data.columns if col not in ["Close", "Open-orig", "High-orig", "Low-orig", "Close-orig", "Ticker"]]
    //self.astro_feature_columns = [col for col in self.full_data.columns if col not in ['MACD', 'Signal', 'Hist', 'High', 'Low', 'Open', "Close", "Open-orig", "High-orig", "Low-orig", "Close-orig", "Ticker"]]
    //self.tech_feature_columns = [col for col in self.full_data.columns if col in ['MACD', 'Signal', 'Hist', 'High', 'Low', 'Open']]
    //self.state_dim = len(self.feature_columns) + 2  + self.risk_feature_dim
        
    std::vector<float> price_window;
    std::vector<float>feature_window;
    std::vector<float> tech_window;
    std::vector<float>returns_window;
    //std::vector<__DATE__>date_window;
        
    int window_position = 0;
    int returns_position = 0;
    bool windows_filled = false;
        
    std::random_device rd;
    
    float sharpe = 0.0;
    float volatility = 0.0;
    float rel_strength = 0.0;
    float reward_val = 0.0;
        
    //std::vector<float, float, float, float, float, float, std::string, int, float, bool> info;

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

    float calculate_local_sharpe(const std::vector<float>& returns_vector,int window_lookback) {
        float risk_free_rate = 0.02f;
    
        if (returns_vector.size() < window_lookback) {
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

    float calculate_relative_strength(const std::vector<float>& prices, int current_idx, int window_lookback) {
        if (current_idx < window_lookback) {
            return 0.0f;
        }

        float current_price = prices[current_idx];
        int start_idx = std::max(0, current_idx - window_lookback);

        float sum = std::accumulate(prices.begin() + start_idx, prices.begin() + current_idx, 0.0f);
        float avg_price = sum / (current_idx - start_idx);
    
        return (current_price - avg_price) / avg_price;
    }

    int action_space(int possible_actions){

        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distrib(1, possible_actions);
        int random_number = distrib(gen);
        return random_number;
    }

public:
    
    std::tuple<float, float, float> calculate_pnl_metrics(
        const float current_price, 
        const float entry_price, 
        const float trading_fee, 
        const float portfolio_value, 
        const std::vector<float>& returns_vector
    ) {
        float pnl_pct = ((current_price - entry_price) / (std::abs(entry_price) + 0.00001f));
        float trade_cost_pct = (trading_fee * 2) / portfolio_value;
        float net_pnl_pct = pnl_pct - trade_cost_pct;
        float risk_adjusted_pnl = 0.0f;

        if (returns_vector.size() > 1) {
            float volatility = calculate_std(returns_vector);
            volatility = clip(volatility, 0.001f, 0.5f);
            risk_adjusted_pnl = net_pnl_pct / (volatility + 1e-4f);
        } else {
            risk_adjusted_pnl = net_pnl_pct;
        }

        return std::make_tuple(trade_cost_pct, net_pnl_pct, risk_adjusted_pnl);
    }

    std::tuple<float, float> calculate_risk_metrics(
        const std::vector<float>& prices,
        const int current_idx,
        const int window_lookback,
        const std::vector<float>& returns_vector,
        int relative_strength_lookback
    ) {
        float sharpe = 0.0f;
        float relative_strength = 0.0f;

        if (returns_vector.size() >= window_lookback) {
            sharpe = calculate_local_sharpe(returns_vector, window_lookback);
        }

        if (prices.size() > current_idx && current_idx >= relative_strength_lookback) {
            relative_strength = calculate_relative_strength(prices, current_idx, relative_strength_lookback);
        }

        return std::make_tuple(sharpe, relative_strength);
    }
};

PYBIND11_MODULE(MarketEnvCpp, m) {
    py::class_<MarketEnv>(m, "MarketEnv")
        .def(py::init<>())
        .def("calculate_pnl_metrics", &MarketEnv::calculate_pnl_metrics, "Calculate PnL metrics")
        .def("calculate_risk_metrics", &MarketEnv::calculate_risk_metrics, "Calculate risk metrics");
}