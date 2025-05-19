import pandas as pd
import numpy as np
from numba import jit
from typing import Tuple
import cProfile
import pstats
from classes.CircularBuffer import CircularBuffer
     
@jit(nopython=True, cache=True)
def normalize_reward(reward):
    if abs(reward) > 1:
        return np.sign(reward) * np.log(1 + abs(reward))
    return reward

@jit(nopython=True, cache=True)
def calculate_local_sharpe(returns: np.ndarray, lookback: int) -> float:
    if len(returns) < lookback:
        return 0.0
    recent_returns = returns[-lookback:]
    avg_return = np.mean(recent_returns) * 252
    std_return = np.std(recent_returns) * np.sqrt(252)
    if std_return == 0:
        return 0.0
    return (avg_return - 0.02) / std_return

@jit(nopython=True, cache=True)
def calculate_relative_strength(prices: np.ndarray, current_idx: int, window: int) -> float:
    if current_idx < window:
        return 0.0
    current_price = prices[current_idx]
    window_prices = prices[max(0, current_idx-window):current_idx]
    if len(window_prices) == 0:
        return 0.0
    avg_price = np.mean(window_prices)
    return (current_price - avg_price) / avg_price

@jit(nopython=True, cache=True)
def calculate_pnl_metrics(current_price, entry_price, trading_fee, portfolio_value, returns_window):
    pnl_pct = ((current_price - entry_price) / (abs(entry_price) + 1e-2))
    trade_cost_pct = (trading_fee * 2) / portfolio_value
    net_pnl_pct = pnl_pct - trade_cost_pct
    if len(returns_window) > 1:
        volatility = np.std(returns_window)
        volatility = max(0.001, min(volatility, 0.5))
        risk_adjusted_pnl = net_pnl_pct / (volatility + 1e-2)
    else:
        risk_adjusted_pnl = net_pnl_pct
    return trade_cost_pct, net_pnl_pct, risk_adjusted_pnl

class MarketEnv:
    def __init__(
        self, data, initial_capital, max_trades_per_month, 
        trading_fee, hold_penalty, max_hold_steps, segment_size, num_projected_days
    ):
        self.full_data            = data
        self.initial_capital      = initial_capital
        self.max_trades_per_month = max_trades_per_month
        self.segment_size         = segment_size
        self.trading_fee          = trading_fee
        self.hold_penalty         = hold_penalty
        self.max_hold_steps       = max_hold_steps
        self.num_projected_days   = num_projected_days

        self.RETURNS_WINDOW_SIZE  = 30

        # feature setup
        self.feature_columns       = [c for c in data.columns if c not in ["Close","Open-orig","High-orig","Low-orig","Close-orig","Ticker"]]
        self.astro_feature_columns = [c for c in data.columns if c not in ['MACD','Signal','Hist','High','Low','Open','Close','Open-orig','High-orig','Low-orig','Close-orig','Ticker']]
        self.tech_feature_columns  = [c for c in data.columns if c in ['MACD','Signal','Hist','High','Low','Open']]
        self.num_feature_columns   = len(self.feature_columns)
        self.num_astro_features    = len(self.astro_feature_columns)
        self.num_tech_features     = len(self.tech_feature_columns)
        self.lookback              = 30
        self.risk_feature_dim      = 3
        self.state_dim             = self.num_feature_columns + 2 + self.risk_feature_dim
        self.action_space          = self.ActionSpace(3)
        
        # circular buffers for rolling windows
        self.price_buf   = CircularBuffer(self.segment_size)
        self.feature_buf = CircularBuffer(self.segment_size, self.num_feature_columns)
        self.returns_buf = CircularBuffer(self.RETURNS_WINDOW_SIZE)

        # segment handling
        self.segments          = self.create_segments()
        self.shuffled_segments = []
        self.shuffle_segments()

        # risk & reward placeholders
        self.annual_return = np.sqrt(252)
        self.sharpe        = 0.0
        self.volatility    = 0.0
        self.rel_strength  = 0.0
        self.reward_val    = 0.0

        self.info = {
            'portfolio_value': None, 'current_price': None, 'open': None,
            'high': None, 'low': None, 'close': None, 'date': None,
            'action_taken': None, 'cash': None, 'holding': None,
        }
        
        self.close_index    = self.full_data.columns.get_loc("Close")
        self.orig_close     = self.full_data.columns.get_loc("Close-orig")
        self.orig_open      = self.full_data.columns.get_loc("Open-orig")
        self.orig_high      = self.full_data.columns.get_loc("High-orig")
        self.orig_low       = self.full_data.columns.get_loc("Low-orig")


    class ActionSpace:
        def __init__(self, n): 
            self.n = n
        def sample(self):
            return np.random.randint(self.n)

    class Box:
        def __init__(self, low, high):
            self.low, self.high, self.shape = low, high, low.shape

    def create_segments(self):
        segs = []
        for t, df in self.full_data.groupby("Ticker"):
            cnt = len(df) // self.segment_size
            for i in range(cnt):
                seg = df.iloc[i*self.segment_size:(i+1)*self.segment_size].copy()
                if len(seg)==self.segment_size: segs.append(seg)
        return segs

    def shuffle_segments(self):
        if not self.shuffled_segments:
            self.shuffled_segments = self.segments.copy()
            np.random.shuffle(self.shuffled_segments)

    def reset(self):
        if not self.shuffled_segments: self.shuffle_segments()
        self.data      = self.shuffled_segments.pop()
        self.data_np = self.data.to_numpy()
        self.data_len  = len(self.data)
        # fresh buffers & state
        self.price_buf   = CircularBuffer(self.segment_size)
        self.feature_buf = CircularBuffer(self.segment_size, self.num_feature_columns)
        self.returns_buf = CircularBuffer(self.RETURNS_WINDOW_SIZE)
        self.current_step     = 0
        self.cash             = self.initial_capital
        self.holding          = False
        self.entry_price      = 0.0
        self.current_month    = self.data.index[0].month
        self.portfolio_value  = self.cash
        self.consecutive_holds= 0
        self.trades_per_month = 0
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        #profiler = cProfile.Profile()
        #profiler.enable()
        done = self.current_step >= self.data_len - 1

        #print(type(self.data))
        # grab prices & features
        norm_price     = self.data_np[self.current_step][self.close_index]
        orig_close     = self.data_np[self.current_step][self.orig_close]
        orig_open      = self.data_np[self.current_step][self.orig_open]
        orig_high      = self.data_np[self.current_step][self.orig_high]
        orig_low       = self.data_np[self.current_step][self.orig_low]
        feats          = self.data_np[self.current_step][:self.num_feature_columns]
        current_date   = self.data.index[self.current_step]

        # compute & insert return if we have history
        if self.price_buf.head > 0:
            prev_price = self.price_buf.latest()
            rtn = ((norm_price - prev_price)/prev_price) if prev_price!=0 else 0.0
            self.returns_buf.insert(rtn)

        # insert new roll-window data
        self.price_buf.insert(norm_price)
        self.feature_buf.insert(feats)

        # volatility for reward calc
        arr = self.returns_buf.get_ordered()
        nz  = arr[arr!=0]
        self.volatility = np.std(nz) if nz.size>0 else 0.01
        self.volatility = np.clip(self.volatility, 0.001, 0.8)

        # terminal check
        if done:
            self.portfolio_value = self.cash + orig_close*float(self.holding)
            self.reward_val = 0.0
            ns = self.get_state()
            self.info.update({
                'portfolio_value': self.portfolio_value,
                'current_price': orig_close,
                'action_taken': "Terminal",
                'cash': self.cash,
                'holding': self.holding
            })
            return ns, self.reward_val, done, self.info

        # reset monthly trades on new month
        if current_date.month != self.current_month:
            self.trades_per_month = 0
            self.current_month    = current_date.month

        # action logic (Buy=1, Sell=2, Hold=0)
        self.reward_val  = 0.0
        action_taken     = None
        sum_vol = 1+self.volatility
        if action == 1:  # Buy
            if self.holding == False:
                self.holding = True
                self.cash -= (orig_close + self.trading_fee)
                self.entry_price = orig_close
                self.trades_per_month += 1
                cost_pct = self.trading_fee / self.portfolio_value
                self.reward_val = -cost_pct
            else:
                if self.cash < (orig_close + self.trading_fee):
                    self.reward_val = -2.5*sum_vol
                elif self.trades_per_month >= self.max_trades_per_month:
                    self.reward_val = -1.5*sum_vol
                else:
                    self.reward_val = -3.5*sum_vol

        elif action == 2:  # Sell
            if not self.holding:
                self.reward_val = -2.5*(1+self.volatility)
                action_taken = "Invalid Sell - Not Holding"
            elif self.trades_per_month >= self.max_trades_per_month:
                self.reward_val = -1.5*(1+self.volatility)
                action_taken = "Invalid Sell - Trade Limit Exceeded"
            else:
                cost_pct, net_pct, risk_adj = calculate_pnl_metrics(
                    orig_close,
                    self.entry_price,
                    self.trading_fee,
                    self.portfolio_value,
                    self.returns_buf.get_ordered()
                )
                self.cash += (orig_close - self.trading_fee)
                self.holding = False
                self.trades_per_month += 1
                self.portfolio_value = self.cash
                hold_opt = max(1, int(10*(1-self.volatility)))
                dur_fac  = np.exp(-0.5*((self.consecutive_holds-hold_opt)/hold_opt)**2)
                self.reward_val = risk_adj * dur_fac
                self.reward_val *= 1.1 if net_pct>0 else 0.9
                self.consecutive_holds = 0
                action_taken = "Sell"

        else:  # Hold
            if self.holding:
                cost_pct, net_pct, risk_adj = calculate_pnl_metrics(
                    orig_close,
                    self.entry_price,
                    self.trading_fee,
                    self.portfolio_value,
                    self.returns_buf.get_ordered()
                )
                self.reward_val = 0.1 * net_pct
                penalty = self.hold_penalty * (1.1**min(self.consecutive_holds,20))
                self.reward_val -= penalty
                max_h = max(1, int(self.max_hold_steps*(1+self.volatility)))
                if self.consecutive_holds > max_h:
                    self.reward_val -= 0.05*(self.consecutive_holds-max_h)
                action_taken = "Hold - Holding Position"
            else:
                self.reward_val = -self.hold_penalty*(1.2**min(self.consecutive_holds,12))
                action_taken = "Hold - No Position"
            self.consecutive_holds += 1

        # finalize
        self.portfolio_value = self.cash + orig_close*float(self.holding)
        self.reward_val      = normalize_reward(self.reward_val)
        self.reward_val      = np.clip(0.0 if np.isnan(self.reward_val) else self.reward_val, -1.0, 10.0)
        self.current_step   += 1

        next_state = self.get_state()
        self.info.update({
            'portfolio_value': self.portfolio_value,
            'current_price': orig_close,
            'open': orig_open,
            'high': orig_high,
            'low': orig_low,
            'close': orig_close,
            'date': str(current_date),
            'action_taken': action_taken,
            'cash': self.cash,
            'holding': self.holding
        })

        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats("cumtime")
        #stats.print_stats(40)
        return next_state, self.reward_val, done, self.info

    def calculate_risk_metrics(self):
        returns_arr = self.returns_buf.get_ordered()
        price_arr   = self.price_buf.get_ordered()
        self.sharpe       = 0.0
        self.volatility   = 0.0
        self.rel_strength = 0.0

        if len(returns_arr) >= self.lookback:
            w = returns_arr[-self.lookback:]
            self.sharpe     = calculate_local_sharpe(w, self.lookback)
            self.volatility = np.std(w) * self.annual_return
        if len(price_arr) > 0:
            idx = len(price_arr) - 1
            self.rel_strength = calculate_relative_strength(price_arr, idx, self.lookback)

        return self.sharpe, self.volatility, self.rel_strength

    def get_state(self):
        try:
            self.calculate_risk_metrics()
            risk_state      = np.array([self.sharpe, self.volatility, self.rel_strength])
            portfolio_state = np.array([float(self.holding), self.cash/self.initial_capital])

            price_arr   = self.price_buf.get_ordered()
            feature_arr = self.feature_buf.get_ordered()
            n           = price_arr.shape[0]

            if n>0:
                astro_end = n + self.num_projected_days
                astro     = self.data[self.astro_feature_columns].iloc[:astro_end].to_numpy()
            else:
                astro = np.zeros((self.num_projected_days, self.num_astro_features))

            if n>0:
                tech = self.data[self.tech_feature_columns].iloc[:n].to_numpy()
            else:
                tech = np.zeros((1, self.num_tech_features))

            a_len = astro.shape[0]
            t_len = tech.shape[0]
            tgt   = max(a_len, t_len, self.segment_size)

            astro_p = np.pad(astro, ((0, tgt-a_len),(0,0)), 'constant')
            tech_p  = np.pad(tech,  ((0, tgt-t_len),(0,0)), 'constant')
            combo   = np.hstack([tech_p, astro_p])

            p_states = np.tile(portfolio_state, (tgt,1))
            r_states = np.tile(risk_state,     (tgt,1))

            fs = np.column_stack([combo, p_states, r_states])
            if fs.shape[0]>self.segment_size:
                fs = fs[-self.segment_size:,:]

            fs = np.nan_to_num(fs, nan=0.0, posinf=1.0, neginf=-1.0)
            return fs.astype(np.float16)

        except Exception as e:
            print(f"Error in get_state: {e}")
            return np.zeros((self.segment_size, self.state_dim), dtype=np.float16)
