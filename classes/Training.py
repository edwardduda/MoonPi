import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import cProfile
import pstats

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

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        s_batch  = torch.as_tensor(states,       device=self.device)        # (B,70,179)
        ns_batch = torch.as_tensor(next_states,  device=self.device)
        a_batch  = torch.as_tensor(actions,      device=self.device).unsqueeze(1)  # (B,1)
        r_batch  = torch.as_tensor(rewards,      device=self.device).unsqueeze(1)  # (B,1)
        d_batch  = torch.as_tensor(dones.astype(np.float32), device=self.device).unsqueeze(1)

        q_curr, (tech_w, temp_w, astro_w) = self.main_model(s_batch)        # (B,nA)
        q_a = q_curr.gather(1, a_batch)                                     # (B,1)
        
        with torch.no_grad():
            q_next_main, _   = self.main_model(ns_batch)
            next_acts        = q_next_main.argmax(dim=1, keepdim=True)      # (B,1)

            q_next_targ, _   = self.target_model(ns_batch)
            q_next           = q_next_targ.gather(1, next_acts)             # (B,1)
            q_next           = q_next * (1.0 - d_batch)                     # zero on terminal
            target_q         = r_batch + self.gamma * q_next

        loss = F.smooth_l1_loss(q_a, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 1.05)
        self.optimizer.step()
        self.scheduler.step()

        for tgt, src in zip(self.target_model.parameters(), self.main_model.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)

        self.logger.log_training_step(
            epsilon          = self.get_current_epsilon(),
            lr               = self.scheduler.get_last_lr()[0],
            reward           = r_batch.mean().item(),
            loss             = loss.item(),
            main_q_values    = q_curr,
            target_q_values  = q_next_targ,
            temporal_weights = temp_w,
            feature_weights  = astro_w,      # <-- astrology attention
            technical_weights= tech_w
        )

        self.total_steps += 1
        self.epsilon_schedule.step()
        
        return loss.item()

    def take_action(self, state):

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if len(self.replay_buffer) < self.min_replay_size or \
        random.random() < self.get_current_epsilon():
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals, _ = self.main_model(state_tensor)  # (1,nA)
                action    = q_vals.argmax(dim=1).item()

        next_state, reward, done, _ = self.env.step(action)

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
                _ = self.training_step()

            ep_r += reward

        final_val = self.env.portfolio_value
        self.logger.log_episode_pnl(self.env.initial_capital, final_val)
        self.episodes_done += 1
        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats("cumtime")
        #stats.print_stats(50)
        if self.episodes_done % 150 == 0 and self.max_trades_per_month > 3:
            self.max_trades_per_month -= 1
        if self.episodes_done % 10 == 0:
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
