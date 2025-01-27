from numba import jit
import numpy as np

@jit(nopython=True)
def normalize_reward(reward):
    if abs(reward) > 1:
        return np.sign(reward) * np.log(1 + abs(reward))
    return reward

@jit(nopython=True)
def calculate_pnl_metrics(current_price, entry_price, trading_fee, portfolio_value, returns_window, position_size=1.0):
    # Calculate raw PnL percentage
    pnl_pct = ((current_price - entry_price) / (abs(entry_price) + 1e-4)) * position_size
        
    # Calculate trade costs
    trade_cost_pct = (trading_fee * 2) / portfolio_value
        
    # Net PnL after costs
    net_pnl_pct = pnl_pct - trade_cost_pct
        
    # Calculate risk-adjusted PnL using volatility from returns window
    if len(returns_window) > 1:
        volatility = np.std(returns_window)
        # Replace np.clip with min and max
        volatility = max(0.001, min(volatility, 0.5))  # Manually clip volatility between 0.001 and 0.5
        risk_adjusted_pnl = net_pnl_pct / (volatility + 1e-8)
    else:
        risk_adjusted_pnl = net_pnl_pct
            
    return {
        'raw_pnl_pct': pnl_pct,
        'trade_cost_pct': trade_cost_pct,
        'net_pnl_pct': net_pnl_pct,
        'risk_adjusted_pnl': risk_adjusted_pnl
    }

@jit(nopython=True)
def calculate_local_sharpe(returns: np.ndarray, lookback: int) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) < lookback:
        return 0.0
    
    recent_returns = returns[-lookback:]
    avg_return = np.mean(recent_returns) * 252
    std_return = np.std(recent_returns) * np.sqrt(252)
    
    if std_return == 0:
        return 0.0
        
    return (avg_return - 0.02) / std_return

@jit(nopython=True)
def get_low_vol_penalty(volatility):
    base_penalty = 0.005  # 0.5% base penalty
    vol_threshold = 0.15  # 15% annualized volatility threshold
    
    if volatility < vol_threshold:
        # Exponentially increase penalty as volatility decreases
        penalty_multiplier = np.exp((vol_threshold - volatility) * 10)
        return base_penalty * penalty_multiplier
    return 0.0

@jit(nopython=True)
def calculate_relative_strength(prices: np.ndarray, current_idx: int, window: int = 40) -> float:
    """Calculate relative strength indicator"""
    if current_idx < window:
        return 0.0
        
    current_price = prices[current_idx]
    window_prices = prices[max(0, current_idx-window):current_idx]
    
    if len(window_prices) == 0:
        return 0.0
        
    avg_price = np.mean(window_prices)
    
    return (current_price - avg_price) / avg_price

class MarketEnvJit():
    def __init__(self, lookback : int):
        self.lookback = lookback
        
        pass
    
    def calculate_pnl_metrics(self, current_price, entry_price, trading_fee, portfolio_value, returns_window, position_size=1.0):
        return calculate_pnl_metrics(self, current_price, entry_price, trading_fee, portfolio_value, returns_window, position_size)
    
    def calculate_local_sharpe(self, returns, lockback):
        return calculate_local_sharpe(returns, lockback)
    
    def calculate_local_sharpe(self, returns, lookback):
        return calculate_local_sharpe(returns, lookback)
    
    def calculate_relative_strength(prices, current_idx, window):
        return calculate_relative_strength(prices, current_idx, window)
    
    def normalized_reward(self, reward):
        return normalize_reward(reward)
    
