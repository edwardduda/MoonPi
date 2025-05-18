import json
from collections import defaultdict
import numpy as np
import os
from datetime import datetime
import torch

class EpisodeLogger:
    def __init__(self, log_dir="./public/episode_logs"):  # Updated path
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created/verified log directory at: {os.path.abspath(log_dir)}")  # Debug line
        self.reset()
        
    def reset(self):
        """Reset the logger for a new episode"""
        self.episode_data = []
        self.current_step = 0
        self.training_metrics = []
    
    def log_training_step(self, state_batch, action_batch, reward_batch, q_values, loss, time_step):
        """Log training metrics (not used for visualization but kept for analysis)"""
        with torch.no_grad():
            training_data = {
                'time_step': time_step,
                'loss': float(loss),
                'mean_q_value': float(q_values.mean()),
                'mean_reward': float(reward_batch.mean())
            }
            self.training_metrics.append(training_data)
            
    def log_step(self, state, action, reward, info):
        """Log environmental step data for visualization"""
        portfolio_value = float(info.get('portfolio_value', 0))
        initial_capital = float(info.get('initial_capital', 0))
        pnl = portfolio_value - initial_capital
        
        step_data = {
            'step': self.current_step,
            'actions': [int(action)],  # Wrap in array as expected by visualizer
            'q_values': info.get('q_values', [0, 0, 0]),  # Default Q-values if not provided
            'rewards': [float(reward)],  # Wrap in array as expected by visualizer
            'state': [[portfolio_value, pnl]],  # Wrap in nested array as expected by visualizer
            'open': info.get('open', 0),  # Log open price
            'high': info.get('high', 0),  # Log high price
            'low': info.get('low', 0),    # Log low price
            'close': info.get('close', 0), # Log close price
            'date': info.get('date', None),
            'sharpe_ratio': info.get('sharpe_ratio', 0),
            'volatility': info.get('volatility', 0),
            'relative_strength': info.get('relative_strength', 0),
            'portfolioValue': portfolio_value,  # Add this line to match what save_episode expects
            'reward': float(reward),  # Add this line to match what get_episode_summary expects
            'action': int(action)  # Add this line to match what get_episode_summary expects 
        }
        
        self.episode_data.append(step_data)
        self.current_step += 1
        
    def save_episode(self, episode_num):
        """Save the episode data to a JSON file"""
        if not self.episode_data:
            return
            
        filename = os.path.join(self.log_dir, f'episode_{episode_num}.json')
        with open(filename, 'w') as f:
            json.dump({
                'episode_num': episode_num,
                'total_steps': self.current_step,
                'final_portfolio_value': self.episode_data[-1]['portfolioValue'],
                'steps': self.episode_data
            }, f, indent=2)
            
        print(f"Episode {episode_num} data saved to {filename}")
        self.reset()
        
    def get_episode_summary(self):
        """Get summary statistics for the episode"""
        if not self.episode_data:
            return {}
            
        rewards = [step['reward'] for step in self.episode_data]
        portfolio_values = [step['portfolioValue'] for step in self.episode_data]
        actions = [step['action'] for step in self.episode_data]
        
        return {
            'total_steps': self.current_step,
            'total_reward': sum(rewards),
            'mean_reward': np.mean(rewards),
            'final_portfolio_value': portfolio_values[-1],
            'max_portfolio_value': max(portfolio_values),
            'min_portfolio_value': min(portfolio_values),
            'action_counts': {
                'hold': actions.count(0),
                'buy': actions.count(1),
                'sell': actions.count(2)
            }
        }
    def save_training_session(self, episode_num, model_metrics=None):
        """Save the episode data with both environmental and training metrics"""
        if not self.episode_data:
            return
            
        # Get absolute path
        abs_path = os.path.abspath(self.log_dir)
        filename = os.path.join(abs_path, f'training_episode_{episode_num}.json')
        
        # Combine environmental and training metrics
        training_data = {
            'episode_num': episode_num,
            'total_steps': len(self.episode_data),
            'metrics': {
                **(model_metrics or {}),
                'training_metrics': {
                    'mean_loss': np.mean([m['loss'] for m in self.training_metrics]) if self.training_metrics else 0,
                    'mean_q_value': np.mean([m['mean_q_value'] for m in self.training_metrics]) if self.training_metrics else 0,
                }
            },
            'steps': self.episode_data
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2)
            print(f"Successfully saved episode data to {filename}")
            print(f"File size: {os.path.getsize(filename)} bytes")
        except Exception as e:
            print(f"Error saving episode data: {e}")
        
        self.reset()  # Reset both episode data and training metrics