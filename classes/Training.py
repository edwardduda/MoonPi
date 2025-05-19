import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import cProfile
import pstats
import gc

from classes.EpisodeLogger    import EpisodeLogger
from classes.DQNLogger        import DQNLogger
from classes.LearningRateScheduler import CosineAnnealingLRScheduler
from classes.EpsilonSchedule  import EpsilonSchedule
from classes.ReplayBuffer     import ReplayBuffer

class Training:
    def __init__(self, env, main_model, target_model, config):
        self.env          = env
        self.main_model   = main_model
        self.target_model = target_model
        self.config       = config

        tp = config.TRAINING_PARMS
        self.episodes        = tp['EPISODES']
        self.buffer_size     = tp['BUFFER_SIZE']
        self.batch_size      = tp['BATCH_SIZE']
        self.gamma           = tp['GAMMA']
        self.tau             = tp['TAU']
        self.device          = tp['DEVICE']
        self.learning_rate   = tp['LEARNING_RATE']
        self.min_replay_size = tp['MIN_REPLAY_SIZE']

        mp = config.MARKET_ENV_PARMS
        self.max_trades_per_month = mp['MAX_TRADES_PER_MONTH']
        self.epsilon_schedule = EpsilonSchedule(
            warmup_steps=10,
            start=tp['EPSILON_START'],
            end=tp['EPSILON_END'],
            reset=tp['EPSILON_RESET'],
            period=tp['STEPS_PER_EPISODE'] * 60,
        )
        self.optimizer = optim.AdamW(
            self.main_model.parameters(),
            lr=self.learning_rate,
            weight_decay=tp['WEIGHT_DECAY']
        )
        self.scheduler = CosineAnnealingLRScheduler(
            optimizer=self.optimizer,
            initial_lr=self.learning_rate,
            min_lr=tp['MIN_LEARNING_RATE'],
            period=tp['STEPS_PER_EPISODE'] * 40
        )

        state_shape = (env.segment_size, env.state_dim)   # e.g. (70, 179)
        print(state_shape)
        self.replay_buffer = ReplayBuffer(state_shape, self.buffer_size)

        self.episode_logger = EpisodeLogger()
        self.logger         = DQNLogger(
            log_dir="/Users/edwardduda/Desktop/MoonPi/runs",
            scalar_freq=config.DATA_CONFIG['SEGMENT_SIZE'],
            attention_freq=config.DATA_CONFIG['SEGMENT_SIZE'],
            histogram_freq=config.DATA_CONFIG['SEGMENT_SIZE'],
            buffer_size=config.DATA_CONFIG['SEGMENT_SIZE'] * 2
        )
        self.episode_bar     = tqdm(range(self.episodes), desc="Episodes")
        self.buffer_bar      = tqdm(range(self.min_replay_size), desc="Filling Buffer")

        self.total_steps   = 0
        self.episodes_done = 0
         
    def get_current_epsilon(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return self.epsilon_schedule.start
        return self.epsilon_schedule.eps

    def training_step(self):
        # Configure microbatches
        micro_batch_size = 4
        num_micro_batches = self.batch_size // micro_batch_size
        
        # Sample full batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        # Save last attention weights for logging
        saved_weights = None
        # Save Q-values for logging
        saved_q_curr = None
        saved_q_targ = None
        
        for i in range(num_micro_batches):
            # Calculate indices
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            
            # Create tensors for this microbatch only
            with torch.no_grad():  # No need to track history for tensor creation
                s_batch = torch.as_tensor(states[start_idx:end_idx], device=self.device)
                ns_batch = torch.as_tensor(next_states[start_idx:end_idx], device=self.device)
                a_batch = torch.as_tensor(actions[start_idx:end_idx], device=self.device).unsqueeze(1)
                r_batch = torch.as_tensor(rewards[start_idx:end_idx], device=self.device).unsqueeze(1)
                d_batch = torch.as_tensor(dones[start_idx:end_idx].astype(np.float32), device=self.device).unsqueeze(1)
            
            # Forward pass
            q_curr, weights = self.main_model(s_batch)
            tech_w, temp_w, astro_w = weights
            saved_weights = (tech_w, temp_w, astro_w)  # Save for logging
            
            # Save Q-values from the last batch for logging
            if i == num_micro_batches - 1:
                saved_q_curr = q_curr.detach().clone()
            
            q_a = q_curr.gather(1, a_batch)
            
            # Target calculation
            with torch.no_grad():
                q_next_main, _ = self.main_model(ns_batch)
                next_acts = q_next_main.argmax(dim=1, keepdim=True)
                
                q_next_targ, _ = self.target_model(ns_batch)
                
                # Save target Q-values from the last batch for logging
                if i == num_micro_batches - 1:
                    saved_q_targ = q_next_targ.detach().clone()
                    
                q_next = q_next_targ.gather(1, next_acts)
                q_next = q_next * (1.0 - d_batch)
                target_q = r_batch + self.gamma * q_next
            
            # Scaled loss
            loss = F.smooth_l1_loss(q_a, target_q) / num_micro_batches
            
            # Accumulate loss value for logging (as a float, not tensor)
            total_loss += loss.item() * num_micro_batches
            
            # Backward
            loss.backward()
            
            # Critical: aggressively clear memory
            del s_batch, ns_batch, a_batch, r_batch, d_batch
            del q_curr, q_a, q_next_main, next_acts, q_next_targ, q_next, target_q, loss
            
            # Force garbage collection
            gc.collect()
            
            # Clear MPS cache
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Apply gradients
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Target network update
        for tgt, src in zip(self.target_model.parameters(), self.main_model.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)
        print(f"Training step {self.total_steps}: Q-values logged - main mean: {saved_q_curr.mean().item():.4f}, target mean: {saved_q_targ.mean().item():.4f}")

        # Log using saved info - now with actual Q-values
        self.logger.log_training_step(
            epsilon=self.get_current_epsilon(),
            lr=self.scheduler.get_last_lr()[0],
            reward=np.mean(rewards),  # Use numpy mean on original array
            loss=total_loss,
            main_q_values=saved_q_curr,  # Now passing actual Q-values
            target_q_values=saved_q_targ,  # Now passing actual target Q-values
            temporal_weights=saved_weights[1],
            feature_weights=saved_weights[2],
            technical_weights=saved_weights[0]
        )
        
        self.total_steps += 1
        self.epsilon_schedule.step()
        
        return total_loss

    def take_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if len(self.replay_buffer) < self.min_replay_size or \
        random.random() < self.get_current_epsilon():
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals, _ = self.main_model(state_tensor)  # (1,nA)
                action    = q_vals.argmax(dim=1).item()

        next_state, reward, done, info = self.env.step(action)
        
        # Add this line to log each step
        self.episode_logger.log_step(state, action, reward, info)

        self.replay_buffer.push(state, action, reward, next_state, done)

        replay_buffer_len = len(self.replay_buffer)
        if replay_buffer_len <= self.min_replay_size:
            self.buffer_bar.update(1)
            if replay_buffer_len == self.min_replay_size:
                self.buffer_bar.close()
                print("Replay buffer filled — starting training!")

        return reward, done, next_state

    def episode(self):
        #profiler = cProfile.Profile()
        #profiler.enable()
        state = self.env.reset()
        done  = False
        ep_r  = 0.0

        while not done:
            reward, done, state = self.take_action(state)

            if len(self.replay_buffer) >= self.min_replay_size:
                loss = self.training_step()
                
                # Optionally log the training step if you want detailed metrics
                if len(self.replay_buffer) >= self.min_replay_size:
                    # Get latest batches from training
                    states, actions, rewards, _, _ = self.replay_buffer.sample(self.batch_size)
                    q_values, _ = self.main_model(torch.tensor(states, device=self.device))
                    self.episode_logger.log_training_step(states, actions, rewards, q_values, loss, self.total_steps)

            ep_r += reward

        final_val = self.env.portfolio_value
        self.logger.log_episode_pnl(self.env.initial_capital, final_val)
        self.episodes_done += 1
        
        # Add this line to save the episode data
        model_metrics = {
            "epsilon": self.get_current_epsilon(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "final_portfolio_value": final_val,
            "initial_capital": self.env.initial_capital
        }
        self.episode_logger.save_training_session(self.episodes_done, model_metrics)
        
        if self.episodes_done % 1 == 0:
            self.logger.flush_to_tensorboard()
        
        return ep_r

    def train(self, should_exit=None):
        try:
            for ep in self.episode_bar:
                if should_exit and should_exit():
                    print("Early exit requested.")
                    break
                ep_reward = self.episode()
                self.episode_bar.set_description(f"Ep {ep} | R: {ep_reward:.2f} | ε: {self.get_current_epsilon():.3f}")
            return self.main_model
        except Exception as e:
            print("Error in training loop:", e)
            return self.main_model
        finally:
            if hasattr(self, 'logger'):
                self.logger.close()
