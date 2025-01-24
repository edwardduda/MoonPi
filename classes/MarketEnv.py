import pandas as pd
import numpy as np
from numba import jit
from typing import Tuple
from classes.MarketEnvJit import MarketEnvJit

@jit(nopython=True)
def normalize_reward(reward):
    if abs(reward) > 1:
        return np.sign(reward) * np.log(1 + abs(reward))
    return reward

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
def calculate_local_sharpe(returns: np.ndarray, lookback) -> float:
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

@jit(nopython=True)
def calculate_pnl_metrics(current_price, entry_price, trading_fee, portfolio_value, returns_window, position_size=1.0):
    # Calculate raw PnL percentage
    pnl_pct = ((current_price - entry_price) / (abs(entry_price) + 1e-8)) * position_size
        
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

class MarketEnv:
    def __init__(self, data, initial_capital, max_trades_per_month, 
                trading_fee, hold_penalty, max_hold_steps, segment_size):
        
        self.full_data = data
        self.initial_capital = initial_capital
        self.max_trades_per_month = max_trades_per_month
        self.segment_size = segment_size
        self.trading_fee = trading_fee
        self.hold_penalty = hold_penalty
        self.max_hold_steps = max_hold_steps
        
        # Constants for window sizes
        self.RETURNS_WINDOW_SIZE = 20
        
        self.action_space = self.ActionSpace(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Add risk metric windows
        self.SHARPE_WINDOW = 20  # For calculating rolling Sharpe ratio
        self.VOLATILITY_WINDOW = 20  # For calculating rolling volatility
        self.sharpe_window = np.zeros(self.SHARPE_WINDOW)
        self.volatility_window = np.zeros(self.VOLATILITY_WINDOW)
        self.risk_feature_dim = 3 
        
        self.risk_metrics = (0.0, 0.0, 0.0)
        
        # Get feature dimensions (excluding Close and Ticker)
        self.feature_columns = [col for col in self.full_data.columns if col not in ["Close", "Open-orig", "High-orig", "Low-orig", "Close-orig", "Ticker"]]
        self.state_dim = len(self.feature_columns) + 2  + self.risk_feature_dim
        
        # Initialize numpy arrays for windows
        self.price_window = np.zeros(self.segment_size)
        self.feature_window = np.zeros((self.segment_size, len(self.feature_columns)))
        self.returns_window = np.zeros(self.RETURNS_WINDOW_SIZE)
        self.date_window = np.zeros(self.segment_size, dtype='datetime64[ns]')
        
        # Track current position in windows
        self.window_position = 0
        self.returns_position = 0
        self.windows_filled = False
        
        # Create segments
        self.segments = self.create_segments()
        self.shuffled_segments = []
        self.shuffle_segments()
        
        # Initialize observation space
        high = np.inf * np.ones((self.segment_size, self.state_dim))
        low = -np.inf * np.ones((self.segment_size, self.state_dim))
        self.observation_space = self.Box(low=low, high=high)
    
    class ActionSpace:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return np.random.randint(self.n)

    class Box:
        def __init__(self, low, high):
            self.low = low
            self.high = high
            self.shape = low.shape
            
    def update_window(self, window, value, position):
        """Helper method to update rolling windows"""
        if not self.windows_filled:
            window[position] = value
        else:
            # Roll the window and update the newest value
            window[:-1] = window[1:]
            window[-1] = value
            
    def create_segments(self):
        segments = []
        for ticker, stock_data in self.full_data.groupby("Ticker"):
            num_segments = len(stock_data) // self.segment_size
            for i in range(num_segments):
                segment = stock_data.iloc[i * self.segment_size:(i + 1) * self.segment_size].copy()
                if len(segment) == self.segment_size:
                    segments.append(segment)
        return segments
    
    def update_feature_window(self, features, position):
        """Helper method to update feature window"""
        if not self.windows_filled:
            self.feature_window[position] = features
        else:
            # Roll the window and update the newest value
            self.feature_window[:-1] = self.feature_window[1:]
            self.feature_window[-1] = features
            
    def calculate_risk_metrics(self, current_price):
        """Calculate various risk metrics for the current state"""
        
        # Calculate rolling Sharpe ratio
        if len(self.returns_window) >= self.SHARPE_WINDOW:
            recent_returns = self.returns_window[-self.SHARPE_WINDOW:]
            sharpe = calculate_local_sharpe(recent_returns, self.SHARPE_WINDOW)
        else:
            sharpe = 0.0
            
        # Calculate rolling volatility
        if len(self.returns_window) >= self.VOLATILITY_WINDOW:
            recent_returns = self.returns_window[-self.VOLATILITY_WINDOW:]
            volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0.0
            
        # Calculate relative strength
        if self.window_position > 0 or self.windows_filled:
            relative_strength = calculate_relative_strength(
                self.price_window, 
                self.window_position if not self.windows_filled else len(self.price_window) - 1
            )
        else:
            relative_strength = 0.0
            
        self.risk_metrics = (sharpe, volatility, relative_strength)
        
        return sharpe, volatility, relative_strength
    
    def shuffle_segments(self):
        if not self.shuffled_segments:
            self.shuffled_segments = self.segments.copy()
            np.random.shuffle(self.shuffled_segments)
    
    def get_state(self):
        try:
            # Calculate risk metrics
            self.calculate_risk_metrics(self.price_window[-1])
            
            # Create risk state features
            risk_state = np.array([
                self.risk_metrics[0],
                self.risk_metrics[1],
                self.risk_metrics[0]
            ])
            
            # Add portfolio state features
            portfolio_state = np.array([
                self.cash / self.initial_capital,
                float(self.holding)
            ])
            
            # Combine market features with portfolio and risk state
            if self.window_position > 0 or self.windows_filled:
                valid_features = self.feature_window[:self.window_position] if not self.windows_filled else self.feature_window
                # Tile portfolio and risk states to match feature window length
                portfolio_states = np.tile(portfolio_state, (len(valid_features), 1))
                risk_states = np.tile(risk_state, (len(valid_features), 1))
                # Concatenate all features
                full_state = np.column_stack([valid_features, portfolio_states, risk_states])
            else:
                full_state = np.zeros((self.segment_size, self.state_dim))
            
            # Pad if necessary
            if full_state.shape[0] < self.segment_size:
                pad_size = self.segment_size - full_state.shape[0]
                padding = np.zeros((pad_size, full_state.shape[1]))
                full_state = np.vstack([padding, full_state])
            
            # Handle NaN values and ensure correct type
            full_state = np.nan_to_num(full_state, nan=0.0, posinf=1.0, neginf=-1.0)
            return full_state.astype(np.float32)
            
        except Exception as e:
            print(f"Error in get_state: {e}")
            return np.zeros((self.segment_size, self.state_dim), dtype=np.float32)

    def reset(self):
        if not self.shuffled_segments:
            self.shuffle_segments()

        # Get new segment
        self.data = self.shuffled_segments.pop()
        
        self.epi_norm_close_np = self.data["Close"].to_numpy()  # Normalized price for state
        self.epi_orig_close_np = self.data["Close-orig"].to_numpy()  # Original price for PnL
        self.epi_open_orig_np = self.data["Open-orig"].to_numpy()
        self.epi_high_orig_np= self.data["High-orig"].to_numpy()
        self.epi_low_orig_np = self.data["Low-orig"].to_numpy()
        
        # Reset all window arrays
        self.price_window.fill(0)
        self.feature_window.fill(0)
        self.returns_window.fill(0)
        self.date_window.fill(np.datetime64('NaT'))
        
        # Reset window tracking
        self.window_position = 0
        self.returns_position = 0
        self.windows_filled = False
        
        # Initialize with first values
        initial_price = self.epi_norm_close_np[0]
        initial_features = self.data[self.feature_columns].iloc[0].values
        initial_date = self.data.index[0]
        
        # Update windows with initial values
        self.update_window(self.price_window, initial_price, 0)
        self.update_feature_window(initial_features, 0)
        self.update_window(self.date_window, np.datetime64(initial_date), 0)
        
        # Reset state variables
        self.current_step = 0
        self.cash = self.initial_capital
        self.holding = False
        self.entry_price = 0.0
        self.current_month = initial_date.month
        self.portfolio_value = self.cash
        self.consecutive_holds = 0
        self.trades_per_month = 0

        return self.get_state()
        
    def step(self, action: int):
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get both normalized and original prices
        norm_price = self.epi_norm_close_np[self.current_step]  # Normalized price for state
        original_close = self.epi_orig_close_np[self.current_step] # Original price for PnL
        original_open = self.epi_open_orig_np[self.current_step]
        original_high = self.epi_high_orig_np[self.current_step]
        original_low = self.epi_low_orig_np[self.current_step]
        
        current_features = self.data[self.feature_columns].iloc[self.current_step].values
        current_date = self.data.index[self.current_step]
        
        # Update windows with normalized price for state representation
        next_pos = (self.window_position + 1) % self.segment_size
        self.windows_filled = self.windows_filled or next_pos == 0
        
        self.update_window(self.price_window, norm_price, self.window_position)
        self.update_feature_window(current_features, self.window_position)
        self.update_window(self.date_window, np.datetime64(current_date), self.window_position)
        
        # Calculate returns using normalized prices for state features
        if self.window_position > 0 or self.windows_filled:
            prev_price = self.price_window[self.window_position - 1] if not self.windows_filled else self.price_window[-2]
            returns = ((norm_price - prev_price) / prev_price) if prev_price != 0 else 0.0
            self.update_window(self.returns_window, returns, self.returns_position)
            self.returns_position = (self.returns_position + 1) % self.RETURNS_WINDOW_SIZE
        
        self.window_position = next_pos
        
        # Initialize default values
        reward = 0.0
        action_taken = None

        if done:
            # Calculate final portfolio value using original prices
            self.portfolio_value = self.cash + (original_close * float(self.holding))
            reward = 0.0
            next_state = self.get_state()
            info = {
                'portfolio_value': self.portfolio_value,
                'current_price': original_close,  # Use original price
                'action_taken': "Terminal",
                'cash': self.cash,
                'holding': self.holding
            }
            return next_state, reward, done, info

        # Calculate volatility from returns window
        non_zero_returns = self.returns_window[self.returns_window != 0]
        volatility = np.std(non_zero_returns) if non_zero_returns.size > 0 else 0.01
        volatility = np.clip(volatility, 0.001, 0.5)

        # Check if new month has started
        current_month = current_date.month
        if current_month != self.current_month:
            self.trades_per_month = 0
            self.current_month = current_month

        low_vol_penalty = get_low_vol_penalty(volatility)
        # Execute action using original prices
        if action == 1:  # Buy
            if self.holding:
                reward = -2.5 * (1 + volatility)
                action_taken = "Invalid Buy - Already Holding"
            elif self.cash < (original_close + self.trading_fee):  # Check using original price
                reward = -3.5 * (1 + volatility)
                action_taken = "Invalid Buy - Insufficient Cash"
            elif self.trades_per_month >= self.max_trades_per_month:
                reward = -1.5 * (1 + volatility)
                action_taken = "Invalid Buy - Trade Limit Exceeded"
            else:
                
                current_sharpe = self.risk_metrics[0]
                if current_sharpe > 2.0:  # Excellent conditions
                    sharpe_multiplier = 1.2
                    action_taken = "Buy - Excellent Sharpe"
                elif current_sharpe > 1.0:  # Good conditions
                    sharpe_multiplier = 1.1
                    action_taken = "Buy - Good Sharpe"
                elif current_sharpe >= 0:  # Neutral conditions
                    sharpe_multiplier = 1.0
                    action_taken = "Buy - Neutral Sharpe"
                else:  # Poor conditions
                    sharpe_multiplier = 0.8
                    action_taken = "Buy - Poor Sharpe"
                # Execute Buy with original price
                trade_cost_pct = (self.trading_fee / self.portfolio_value) + low_vol_penalty
                reward = -trade_cost_pct
                self.holding = True
                self.entry_price = original_close  # Store original entry price
                self.cash -= (original_close + self.trading_fee)
                self.trades_per_month += 1

        elif action == 2:  # Sell
            if not self.holding:
                reward = -2.5 * (1 + volatility)
                action_taken = "Invalid Sell - Not Holding"
            elif self.trades_per_month >= self.max_trades_per_month:
                reward = -1.5 * (1 + volatility)
                action_taken = "Invalid Sell - Trade Limit Exceeded"
            else:
                # Calculate base PnL metrics
                pnl_metrics = calculate_pnl_metrics(
                    original_close,
                    self.entry_price,
                    self.trading_fee,
                    self.portfolio_value,
                    self.returns_window
                )
                
                # Check Sharpe ratio conditions for sell decision
                current_sharpe = self.risk_metrics[0]
                if current_sharpe > 2.0:  # Excellent conditions
                    # Penalize early exits in strong trends unless significant profit
                    if pnl_metrics['net_pnl_pct'] > 0.05:  # 5% profit threshold
                        sharpe_multiplier = 1.0  # Neutral - ok to take profit
                        action_taken = "Sell - Profit Taking in Strong Trend"
                    else:
                        sharpe_multiplier = 0.7  # Penalty for early exit
                        action_taken = "Sell - Early Exit in Strong Trend"
                elif current_sharpe > 1.0:  # Good conditions
                    sharpe_multiplier = 1.0
                    action_taken = "Sell - Good Market Conditions"
                elif current_sharpe >= 0:  # Neutral conditions
                    sharpe_multiplier = 1.1  # Slight boost for taking profits
                    action_taken = "Sell - Neutral Market Conditions"
                else:  # Poor conditions
                    sharpe_multiplier = 1.2  # Reward for risk management
                    action_taken = "Sell - Poor Market Conditions"

                # Execute Sell with modified reward
                self.cash += (original_close - self.trading_fee)
                self.holding = False
                self.trades_per_month += 1

                # Base reward using risk-adjusted PnL
                base_reward = pnl_metrics['risk_adjusted_pnl'] - low_vol_penalty
                
                # Apply Sharpe multiplier and duration factors
                optimal_hold_duration = max(1, int(10 * (1 - volatility)))
                duration_factor = np.exp(-0.5 * ((self.consecutive_holds - optimal_hold_duration) / optimal_hold_duration) ** 2)
                reward = base_reward * sharpe_multiplier * duration_factor

                # Additional PnL adjustments
                if pnl_metrics['net_pnl_pct'] > 0:
                    reward *= 1.1
                else:
                    reward *= 0.9

                self.consecutive_holds = 0

        else:  # Hold
            if self.holding:
                # Calculate PnL metrics for current position
                pnl_metrics = calculate_pnl_metrics(
                    original_close,
                    self.entry_price,
                    self.trading_fee,
                    self.portfolio_value,
                    self.returns_window
                )
                
                # Check Sharpe ratio conditions for hold decision
                current_sharpe = self.risk_metrics[0]
                if current_sharpe > 2.0:  # Excellent conditions
                    sharpe_multiplier = 1.3  # Strong reward for holding
                    action_taken = "Hold - Excellent Trend Continuation"
                elif current_sharpe > 1.0:  # Good conditions
                    sharpe_multiplier = 1.1  # Moderate reward
                    action_taken = "Hold - Good Trend Continuation"
                elif current_sharpe >= 0:  # Neutral conditions
                    sharpe_multiplier = 1.0  # Neutral
                    action_taken = "Hold - Neutral Conditions"
                else:  # Poor conditions
                    sharpe_multiplier = 0.7  # Penalty for holding in poor conditions
                    action_taken = "Hold - Poor Conditions"

                # Calculate base reward
                base_reward = 0.1 * pnl_metrics['net_pnl_pct']
                
                # Apply Sharpe multiplier
                reward = base_reward * sharpe_multiplier
                
                # Apply standard hold penalties
                hold_penalty = self.hold_penalty * (1.1 ** min(self.consecutive_holds, 20))
                reward -= hold_penalty

                # Additional penalty for holding too long
                max_optimal_hold = max(1, int(self.max_hold_steps * (1 + volatility)))
                if self.consecutive_holds > max_optimal_hold:
                    reward -= 0.05 * (self.consecutive_holds - max_optimal_hold)
            
            else:  # Not holding
                current_sharpe = self.risk_metrics[0]
                if current_sharpe < 0:  # Poor conditions
                    reward = -0.5 * self.hold_penalty  # Reduced penalty for appropriate waiting
                    action_taken = "Hold - Appropriate Wait in Poor Conditions"
                else:  # Good conditions
                    reward = -self.hold_penalty * (1.2 ** min(self.consecutive_holds, 12))  # Increased penalty
                    action_taken = "Hold - Missing Opportunity"

            self.consecutive_holds += 1

        # Update portfolio value using original price
        self.portfolio_value = self.cash + (original_close * float(self.holding))

        # Normalize and clip reward
        reward = normalize_reward(reward)
        reward = np.clip(0.0 if np.isnan(reward) else reward, -10.0, 10.0)
        
        self.current_step += 1
        next_state = self.get_state()

        info = {
            'portfolio_value': self.portfolio_value,
            'current_price': original_close,  # Use original price in info
            'open': original_open,
            'high': original_high,
            'low': original_low,
            'close': original_close,
            'date': str(current_date),
            'action_taken': action_taken,
            'cash': self.cash,
            'holding': self.holding
        }
        
        return next_state, reward, done, info