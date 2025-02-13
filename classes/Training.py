import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from collections import deque, namedtuple
import random
from classes.EpsilonSchedule import EpsilonSchedule
from classes.DQNLogger import DQNLogger
from classes.LearningRateScheduler import EpsilonMatchingLRScheduler
from classes.EpisodeLogger import EpisodeLogger
import gc

import logging

class Training:
    def __init__(self, env, main_model, target_model, config):
        self.config = config
        self.episode_logger = EpisodeLogger()
        self.env=env
        self.main_model=main_model
        self.target_model=target_model
        self.episodes=config.TRAINING_PARMS.get('EPISODES')
        self.buffer_size=config.TRAINING_PARMS.get('BUFFER_SIZE')
        self.batch_size=config.TRAINING_PARMS.get('BATCH_SIZE')
        self.gamma=config.TRAINING_PARMS.get('GAMMA')
        self.tau=config.TRAINING_PARMS.get('TAU')
        self.learning_rate=config.TRAINING_PARMS.get('LEARNING_RATE')
        self.min_replay_size=config.TRAINING_PARMS.get('MIN_REPLAY_SIZE')
        self.device=config.TRAINING_PARMS.get('DEVICE')
        self.weight_decay=config.TRAINING_PARMS.get('WEIGHT_DECAY')
        self.epsilon_start=config.TRAINING_PARMS.get('EPSILON_START')
        self.epsilon_decay=config.TRAINING_PARMS.get('EPSILON_DECAY')
        self.epsilon_reset=config.TRAINING_PARMS.get('EPSILON_RESET')
        self.epsilon_end=config.TRAINING_PARMS.get('EPSILON_END')
        self.steps_per_episode=config.TRAINING_PARMS.get('STEPS_PER_EPISODE')
        self.initial_capital =config.MARKET_ENV_PARMS.get('INITIAL_CAPITAL')
        self.min_lr=config.TRAINING_PARMS.get('MIN_LEARNING_RATE')
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = DQNLogger(log_dir="/Users/edwardduda/Desktop/MoonPi/runs", scalar_freq=config.DATA_CONFIG.get('SEGMENT_SIZE'), attention_freq=config.DATA_CONFIG.get('SEGMENT_SIZE'), histogram_freq=config.DATA_CONFIG.get('SEGMENT_SIZE'), buffer_size=config.DATA_CONFIG.get('SEGMENT_SIZE') * 2)
        feature_names = []
        feature_names = [col for col in env.feature_columns if col not in ["Close", "Open-orig", "High-orig", "Low-orig", "Close-orig", "Ticker"]]
        feature_names.extend(['Portfolio_Cash', 'Holding_Flag', 'Sharpe_Ratio',
            'Volatility',
            'Relative_Strength'])
        self.logger.feature_names = feature_names
        # Initialize training components
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.optimizer = optim.AdamW(self.main_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = EpsilonMatchingLRScheduler(
            optimizer=self.optimizer,
            initial_lr=self.learning_rate,
            min_lr=self.min_lr,
            warmup_steps=min(self.min_replay_size // 10, 2000),
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_end,
        )
        self.epsilon_schedule = EpsilonSchedule(self.epsilon_start, self.epsilon_end, self.epsilon_decay, self.epsilon_reset)

        self.total_steps = 0
        self.episodes_done = 0

        self.best_reward = float('-inf')
        self.episode_reward = 0
        self.episode_progress_bar = tqdm(range(self.episodes), desc="Training Progress", unit="episode")
        self.replay_buffer_bar = tqdm(range(self.min_replay_size), desc="Replay Buffer", unit=" segments")

        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        
    def get_current_epsilon(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return self.epsilon_start
        else:
            return self.epsilon_schedule.epsilon
        
    def training_step(self):
        transitions = random.sample(self.replay_buffer, self.batch_size)
        batch = self.transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)

        # Get current Q values
        current_q_values, curr_attention = self.main_model(state_batch)
        temporal_weights, feature_weights = curr_attention
        action_q_values = current_q_values.gather(1, action_batch)

        # Calculate target Q values
        with torch.no_grad():
            next_q_values, _ = self.main_model(next_state_batch)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            target_next_q_values, _ = self.target_model(next_state_batch)
            next_q_values = target_next_q_values.gather(1, next_actions)
            target_q_values = reward_batch + self.gamma * next_q_values

        # Calculate loss and optimize
        loss = F.smooth_l1_loss(action_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.1)
        self.optimizer.step()

        # Update target network
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.main_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Log step data
        self.episode_logger.log_training_step(
            state_batch=state_batch,
            action_batch=action_batch,
            reward_batch=reward_batch,
            q_values=current_q_values,
            loss=loss,
            time_step=self.total_steps
        )

        # Original logging
        current_lr = self.scheduler.get_last_lr()[0]
        self.logger.log_training_step(
            epsilon=self.epsilon_schedule.epsilon,
            lr=current_lr,
            reward=torch.mean(reward_batch).item(),
            loss=loss.item(),
            main_q_values=current_q_values,
            target_q_values=target_next_q_values,
            temporal_weights=temporal_weights,
            feature_weights=feature_weights
        )
        
        self.logger.log_feature_importance(feature_weights, self.logger.feature_names)

        self.total_steps += 1
        
        if len(self.replay_buffer) >= self.min_replay_size:
            self.epsilon_schedule.step()
    
    def take_action(self, state, state_tensor):
        if len(self.replay_buffer) < self.min_replay_size or random.random() < self.epsilon_schedule.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values, _ = self.main_model(state_tensor)  # Note: unpack the attention weights
                action = q_values.max(1)[1].item()
            
        next_state, reward, done, info = self.env.step(action)  # Unpack done flag
    
        # Store transition in replay buffer
        self.replay_buffer.append(self.transition(state, action, reward, next_state, done))
        self.replay_buffer_bar.update(1)
    
        if len(self.replay_buffer) == self.min_replay_size:
            self.replay_buffer_bar.close()
        
        return reward, done, next_state  # Return done flag and next_state
    
    def episode(self):
        state = self.env.reset()
        episode_reward = 0
        initial_capital = self.initial_capital
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action and Q-values
            if len(self.replay_buffer) < self.min_replay_size or random.random() < self.epsilon_schedule.epsilon:
                action = self.env.action_space.sample()
                q_values = [0, 0, 0]  # Default Q-values for random actions
            else:
                with torch.no_grad():
                    q_values_tensor, _ = self.main_model(state_tensor)
                    q_values = q_values_tensor.detach().cpu().numpy()[0].tolist()
                    action = q_values_tensor.max(1)[1].item()
            
            # Take action and get next state
            next_state, reward, done, info = self.env.step(action)
            
            sharpe, volatility, rel_strength = self.env.calculate_risk_metrics(info['current_price'])
            # Only log step data if replay buffer is filled
            if len(self.replay_buffer) >= self.min_replay_size:
                self.episode_logger.log_step(
                    state=state,
                    action=action,
                    reward=reward,
                    info={
                        'portfolio_value': info['portfolio_value'],
                        'current_price': info['current_price'],
                        'initial_capital': initial_capital,
                        'q_values': q_values,
                        'open': info.get('open'),  # Ensure open price is logged
                        'high': info.get('high'),  # High price
                        'low': info.get('low'),    # Low price
                        'close': info.get('close'), # Close price
                        'date': info.get('date'),
                        'sharpe_ratio': sharpe,
                        'volatility': volatility,
                        'relative_strength': rel_strength
                    }
                )

            # Store transition in replay buffer
            self.replay_buffer.append(self.transition(state, action, reward, next_state, done))
            
            # Update progress bar for replay buffer filling
            if len(self.replay_buffer) <= self.min_replay_size:
                self.replay_buffer_bar.update(1)
                if len(self.replay_buffer) == self.min_replay_size:
                    self.replay_buffer_bar.close()
                    print("\nReplay buffer filled, starting training...")
            
            if len(self.replay_buffer) >= self.min_replay_size:
                try:
                    self.training_step()
                except Exception as e:
                    print(f'Error, unable to perform training step {e}')
            
            episode_reward += reward
            state = next_state
        
        # Log episode completion data
        final_portfolio_value = self.env.portfolio_value
        self.logger.log_episode_pnl(initial_capital, final_portfolio_value)
        
        # Only save episode data if replay buffer is filled
        if len(self.replay_buffer) >= self.min_replay_size:
            metrics = {
                'final_portfolio_value': final_portfolio_value,
                'episode_reward': episode_reward,
                'epsilon': self.get_current_epsilon(),
                'replay_buffer_size': len(self.replay_buffer)
            }
            self.episode_logger.save_training_session(self.episodes_done, metrics)
            print(f"\nLogged episode {self.episodes_done} (replay buffer ready)")
        
        self.episodes_done += 1
        
        if self.episodes_done % 500 == 0:
            self.logger.flush_to_tensorboard()
        return episode_reward
    def train(self, should_exit_flag=None):
        try:
            for episode_num in self.episode_progress_bar:
                # Check if we should exit
                if should_exit_flag and should_exit_flag():
                    print("\nExiting training loop...")
                    break
                
                episode_reward = self.episode()
                
                # Update progress bar description
                self.episode_progress_bar.set_description(
                    f"Episode {episode_num} - Reward: {episode_reward:.2f} - ε: {self.get_current_epsilon():.3f}"
                )
            
            return self.main_model
        
        except Exception as e:
            print(f"\nError in training loop: {e}")
            return self.main_model
        finally:
            if hasattr(self, 'logger'):
                self.logger.close()