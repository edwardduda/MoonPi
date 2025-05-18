import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import datetime
import gc
import logging
import scipy
from numba import jit
from classes.CircularBuffer import CircularBuffer

# JIT-compiled function for mean and standard deviation calculation
# More efficient than native NumPy operations for our use case
@jit(nopython=True, cache=True)
def running_mean_std(vals):
    acc, acc2 = 0.0, 0.0
    n = len(vals)
    for v in vals:
        acc  += v
        acc2 += v * v
    mean = acc / n
    std  = (acc2 / n - mean ** 2) ** 0.5
    return mean, std

# Helper function for calculating mean of CircularBuffer values
@jit(nopython=True, cache=True)
def circular_mean(buffer, head, is_full):
    # Calculate mean based on active elements in the buffer
    if is_full:
        return np.mean(buffer)
    elif head > 0:
        return np.mean(buffer[:head])
    else:
        return 0.0  # Return 0 if buffer is empty

class DQNLogger:
    def __init__(self, log_dir, scalar_freq, attention_freq, histogram_freq, buffer_size):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f"{log_dir}/{timestamp}")
        self.step = 0
        self.scalar_freq = scalar_freq
        self.attention_freq = attention_freq
        self.histogram_freq = histogram_freq
        self.buffer_size = buffer_size

        # Buffers for online averaging - using CircularBuffer instead of deque
        # CircularBuffer is more memory efficient and utilizes JIT for faster operations
        self.reward_buffer = CircularBuffer(buffer_size)  # 1D buffer for scalar rewards
        self.loss_buffer = CircularBuffer(buffer_size)    # 1D buffer for scalar losses
        self.pnl_history = CircularBuffer(100)            # 1D buffer with size 100 (matches original deque maxlen)
        
        # Q-value buffers using CircularBuffer
        # Using Python dict to match original structure while benefiting from CircularBuffer internally
        self.q_value_buffer = {
            'main': CircularBuffer(buffer_size),
            'target': CircularBuffer(buffer_size)
        }
        
        # Attention averaging buffers (updated each time, but plotting is delayed)
        # These remain as PyTorch tensors as they need tensor operations
        self.temporal_attention_buffer = None
        self.feature_attention_buffer = None
        self.technical_attention_buffer = None
        self.attention_buffer_count = 0
        self.max_attention_samples = 1000  # samples to average over
        
        # Offline log buffers - keeping as original data structures
        # These are not fixed-size buffers and don't benefit from CircularBuffer
        self.offline_scalars = {}      # metric_name -> list of (step, value)
        self.offline_histograms = {}   # metric_name -> list of (step, data)
        self.offline_attention = []    # list of dicts: { 'step', 'temporal', 'feature' }
        self.offline_pnl = []          # list of (step, pnl_percentage)
        self.episode_pnls = []         # list of all episode PnL percentages
        self.offline_feature_importance = []         # list of dicts: { 'step', 'layer', 'feature_importance', 'feature_names' }
        self.offline_feature_importance_heatmap = []   # list of dicts: { 'step', 'feature_weights', 'feature_names' }
        
        # Optionally, store feature names (used for attention heatmaps)
        self.feature_names = None
        
        # Flag to track if Q-values are being logged
        self.q_values_logged = False

    def initialize_attention_buffers(self, temporal_weights, feature_weights, technical_weights):
        """Initialize the attention buffers with proper shapes."""
        if self.temporal_attention_buffer is None:
            self.temporal_attention_buffer = [
                torch.zeros_like(layer_weights[0])
                for layer_weights in temporal_weights
            ]
        if self.feature_attention_buffer is None:
            self.feature_attention_buffer = [
                torch.zeros_like(layer_weights[0])
                for layer_weights in feature_weights
            ]
        if technical_weights is not None and self.technical_attention_buffer is None:
            self.technical_attention_buffer = [
                torch.zeros_like(layer_weights[0])
                for layer_weights in technical_weights
            ]
    
    def update_attention_buffers(self, temporal_weights, feature_weights, technical_weights=None):
        with torch.no_grad():
            self.initialize_attention_buffers(temporal_weights, feature_weights, technical_weights)
            self.attention_buffer_count = min(self.attention_buffer_count + 1, self.max_attention_samples)
            alpha = 1.0 / self.attention_buffer_count
            # Update temporal attention buffers
            for layer_idx, layer_weights in enumerate(temporal_weights):
                if isinstance(layer_weights, tuple):
                    layer_weights = layer_weights[0]
                self.temporal_attention_buffer[layer_idx] = (
                    (1 - alpha) * self.temporal_attention_buffer[layer_idx] +
                    alpha * layer_weights[0]
                )
            # Update feature attention buffers
            for layer_idx, layer_weights in enumerate(feature_weights):
                if isinstance(layer_weights, tuple):
                    layer_weights = layer_weights[0]
                self.feature_attention_buffer[layer_idx] = (
                    (1 - alpha) * self.feature_attention_buffer[layer_idx] +
                    alpha * layer_weights[0]
                )
            # Update technical attention buffers (if provided)
            if technical_weights is not None:
                for layer_idx, layer_weights in enumerate(technical_weights):
                    if isinstance(layer_weights, tuple):
                        layer_weights = layer_weights[0]
                    self.technical_attention_buffer[layer_idx] = (
                        (1 - alpha) * self.technical_attention_buffer[layer_idx] +
                        alpha * layer_weights[0]
                    )
    
    def log_attention_heatmaps(self, temporal_weights, feature_weights, technical_weights=None):
        # Ensure the attention weights are lists.
        if not isinstance(temporal_weights, (list, tuple)):
            temporal_weights = [temporal_weights]
        if not isinstance(feature_weights, (list, tuple)):
            feature_weights = [feature_weights]
        if technical_weights is not None and not isinstance(technical_weights, (list, tuple)):
            technical_weights = [technical_weights]
        
        plt.ioff()
        try:
            valid_weights = (
                all(isinstance(w, torch.Tensor) for w in temporal_weights + feature_weights) and 
                (technical_weights is None or all(isinstance(w, torch.Tensor) for w in technical_weights))
            )
            if valid_weights:
                self.update_attention_buffers(temporal_weights, feature_weights, technical_weights)
                temporal_copy = [buf.clone() for buf in self.temporal_attention_buffer]
                feature_copy = [buf.clone() for buf in self.feature_attention_buffer]
                technical_copy = (
                    [buf.clone() for buf in self.technical_attention_buffer]
                    if technical_weights is not None else None
                )
                self.offline_attention.append({
                    'step': self.step,
                    'temporal': temporal_copy,
                    'feature': feature_copy,
                    'technical': technical_copy
                })
            else:
                logging.warning("Invalid attention weights provided; skipping offline logging.")
        except Exception as e:
            logging.warning(f"Error in log_attention_heatmaps: {e}")
        finally:
            plt.ion()

    def log_metrics(self, metrics):
        """
        Instead of immediately logging scalars to TensorBoard, store them offline.
        """
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.offline_scalars.setdefault(metric_name, []).append((self.step, value))
            elif torch.is_tensor(value) and value.numel() == 1:
                self.offline_scalars.setdefault(metric_name, []).append((self.step, value.item()))
    
    def log_feature_importance(self, feature_weights, feature_names):
        """
        Save raw feature weights and names offline; plotting happens later.
        """
        plt.ioff()
        try:
            if not feature_weights or not feature_names:
                return
            self.feature_names = feature_names  # save for later use in attention plots
            self.len_feature_names = len(feature_names)
            for layer_idx, layer_weights in enumerate(feature_weights):
                with torch.no_grad():
                    avg_weights = layer_weights.detach().mean(dim=0).cpu().numpy()
                    feature_dim = avg_weights.shape[-1]
                    if feature_dim != self.len_feature_names - 11:
                        logging.warning(f"Feature dimension mismatch: got {feature_dim} vs expected {self.len_feature_names}")
                        continue
                    feature_importance = avg_weights.mean(axis=0)
                    self.offline_feature_importance.append({
                        'step': self.step,
                        'layer': layer_idx,
                        'feature_importance': feature_importance,
                        'feature_names': feature_names
                    })
        except Exception as e:
            logging.error(f"Error in log_feature_importance: {e}")
        finally:
            plt.ion()
    
    def log_training_step(self, epsilon, lr, reward, loss, main_q_values, target_q_values, 
                          temporal_weights=None, feature_weights=None, technical_weights=None):
        """
        Store training metrics offline. Histograms and attention data are saved rather than plotted.
        Uses CircularBuffer for efficient storage and computation.
        """
        try:
            # Debug: Print Q-values status
            logging.info(f"Q-values provided: main={main_q_values is not None}, target={target_q_values is not None}")
            
            # Insert values into CircularBuffer instances instead of appending to deques
            if reward is not None:
                self.reward_buffer.insert(reward)
            if loss is not None:
                self.loss_buffer.insert(loss)
            
            # FIXED: Always log Q-values directly when available, regardless of other conditions
            with torch.no_grad():
                if main_q_values is not None:
                    q_mean = main_q_values.mean().item()
                    self.q_value_buffer['main'].insert(q_mean)
                    # Remove the conditional to always log directly
                    self.offline_scalars.setdefault('q_values/direct_main_mean', []).append((self.step, q_mean))
                    logging.info(f"Directly logged main Q-value: {q_mean} at step {self.step}")
                    self.q_values_logged = True
                
                if target_q_values is not None:
                    q_mean = target_q_values.mean().item()
                    self.q_value_buffer['target'].insert(q_mean)
                    # Remove the conditional to always log directly
                    self.offline_scalars.setdefault('q_values/direct_target_mean', []).append((self.step, q_mean))
                    logging.info(f"Directly logged target Q-value: {q_mean} at step {self.step}")
                    self.q_values_logged = True
            
            if self.step % self.scalar_freq == 0:
                # Check if buffers have data using buffer.head or buffer.is_full attributes
                has_reward_data = self.reward_buffer.head > 0 or self.reward_buffer.is_full
                has_loss_data = self.loss_buffer.head > 0 or self.loss_buffer.is_full
                
                # Debug: Print buffer status
                logging.info(f"Step {self.step}: Reward data: {has_reward_data}, Loss data: {has_loss_data}")
                
                if has_reward_data and has_loss_data:
                    # Get ordered data from CircularBuffer for calculations
                    reward_array = self.reward_buffer.get_ordered()
                    loss_array = self.loss_buffer.get_ordered()
                    
                    # Calculate statistics
                    r_mean, r_std = running_mean_std(reward_array)
                    
                    metrics = {
                        'training/reward': r_mean,
                        'training/loss': np.mean(loss_array),
                        'training/reward_std': r_std if len(reward_array) > 1 else 0,
                        'training/loss_std': np.std(loss_array) if len(loss_array) > 1 else 0,
                        'training/epsilon': epsilon,
                        'training/lr': lr
                    }
                    
                    # Add Q-value means if data exists - ORIGINAL CODE
                    if self.q_value_buffer['main'].head > 0 or self.q_value_buffer['main'].is_full:
                        main_q_array = self.q_value_buffer['main'].get_ordered()
                        q_mean = np.mean(main_q_array)
                        metrics['q_values/main_mean'] = q_mean
                        logging.info(f"Added main Q-value mean to metrics: {q_mean}")
                    
                    if self.q_value_buffer['target'].head > 0 or self.q_value_buffer['target'].is_full:
                        target_q_array = self.q_value_buffer['target'].get_ordered()
                        q_mean = np.mean(target_q_array)
                        metrics['q_values/target_mean'] = q_mean
                        logging.info(f"Added target Q-value mean to metrics: {q_mean}")
                    
                    self.log_metrics(metrics)
            
            if self.step % self.histogram_freq == 0:
                with torch.no_grad():
                    if main_q_values is not None:
                        data = main_q_values.cpu().numpy()
                        self.offline_histograms.setdefault('q_values/main_dist', []).append((self.step, data))
                        logging.info(f"Added main Q-value histogram at step {self.step}")
                    
                    if target_q_values is not None:
                        data = target_q_values.cpu().numpy()
                        self.offline_histograms.setdefault('q_values/target_dist', []).append((self.step, data))
                        logging.info(f"Added target Q-value histogram at step {self.step}")
            
            # Log attention data (including technical weights if provided)
            if (temporal_weights is not None and feature_weights is not None and 
                self.step % self.attention_freq == 0):
                self.log_attention_heatmaps(temporal_weights, feature_weights, technical_weights)
            
            self.step += 1
        
        except Exception as e:
            logging.error(f"Error in log_training_step: {e}")
    
    def log_episode_pnl(self, initial_capital, final_portfolio_value):
        """
        Store episode PnL data offline using CircularBuffer for pnl_history.
        The trend plot is compiled later.
        """
        plt.ioff()
        try:
            pnl_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
            
            # Insert into CircularBuffer instead of appending to deque
            self.pnl_history.insert(pnl_percentage)
            
            # Keep appending to the growing list as in the original
            self.episode_pnls.append(pnl_percentage)
            self.offline_scalars.setdefault('performance/episode_pnl_percentage', []).append((self.step, pnl_percentage))
            self.offline_pnl.append((self.step, pnl_percentage))
            
            # Calculate average PnL from CircularBuffer
            pnl_array = self.pnl_history.get_ordered()
            avg_pnl = np.mean(pnl_array) if len(pnl_array) > 0 else 0
            
            print(f"\nEpisode PnL: {pnl_percentage:.2f}% | Avg PnL (last 100): {avg_pnl:.2f}%")
            
            # ADDED: Check if we've logged Q-values and suggest flushing if not
            if not self.q_values_logged:
                logging.warning("No Q-values have been logged yet. Make sure to call flush_to_tensorboard() periodically.")
            
            return pnl_percentage
        
        except Exception as e:
            print(f"Error in PnL logging: {e}")
            return 0
        finally:
            plt.ion()
    
    def log_feature_importance_heatmap(self, feature_weights, feature_names):
        """
        Save raw feature weights and names for later heatmap generation.
        """
        plt.ioff()
        try:
            self.offline_feature_importance_heatmap.append({
                'step': self.step,
                'feature_weights': feature_weights,
                'feature_names': feature_names
            })
        except Exception as e:
            logging.warning(f"Error in log_feature_importance_heatmap: {e}")
        finally:
            plt.ion()
    
    def flush_to_tensorboard(self):
        """
        Dump everything in the offline buffers to TensorBoard, then clear them.
        """
        # Debug message
        logging.info(f"Flushing to TensorBoard at step {self.step}")
        
        # ------------------------------------------------------------------ scalars
        scalar_count = 0
        for metric, entries in self.offline_scalars.items():
            for step, value in entries:
                self.writer.add_scalar(metric, value, step)
                scalar_count += 1
                if 'q_values' in metric:
                    self.q_values_logged = True
                    logging.info(f"Logged Q-value metric to TensorBoard: {metric}={value} at step {step}")

        # ------------------------------------------------------------------ histograms
        histogram_count = 0
        for metric, entries in self.offline_histograms.items():
            for step, data in entries:
                self.writer.add_histogram(metric, data, step)
                histogram_count += 1
                if 'q_values' in metric:
                    self.q_values_logged = True
                    logging.info(f"Logged Q-value histogram to TensorBoard: {metric} at step {step}")

        # ------------------------------------------------------------------ attention heat-maps
        for log in self.offline_attention:
            step = log['step']

            # -------- temporal --------------
            for layer_idx, avg_weights in enumerate(log['temporal']):
                arr = avg_weights.cpu().numpy()
                if arr.ndim > 2:   # heads × L × L  ->  avg over heads
                    arr = arr.mean(axis=0)
                if arr.ndim < 2:
                    continue
                fig = plt.figure(figsize=(36, 24))
                sns.heatmap(arr, cmap='viridis')
                plt.title(f'Temporal Attention (Layer {layer_idx+1})')
                buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0)
                img = Image.open(buf).resize((500,500))
                self.writer.add_image(f'attention/temporal_layer_{layer_idx+1}',
                                    np.array(img).transpose(2,0,1), step)
                plt.close(fig); buf.close()

            # -------- astrology / feature --------------
            for layer_idx, avg_weights in enumerate(log['feature']):
                arr = avg_weights.cpu().numpy()
                if arr.ndim > 2:
                    arr = arr.mean(axis=0)
                if arr.ndim < 2:
                    continue
                fig = plt.figure(figsize=(36, 24))
                sns.heatmap(arr, cmap='viridis',
                            xticklabels=self.feature_names or False,
                            yticklabels=self.feature_names or False)
                plt.title(f'Astrology-Feature Attention (Layer {layer_idx+1})')
                plt.xticks(rotation=45, ha='right')
                buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0)
                img = Image.open(buf).resize((500,500))
                self.writer.add_image(f'attention/feature_layer_{layer_idx+1}',
                                    np.array(img).transpose(2,0,1), step)
                plt.close(fig); buf.close()

            # -------- technical --------------
            for layer_idx, avg_weights in enumerate(log.get('technical', [])):
                arr = avg_weights.cpu().numpy()
                if arr.ndim > 2:
                    arr = arr.mean(axis=0)
                if arr.ndim < 2:
                    continue
                fig = plt.figure(figsize=(24, 36))
                sns.heatmap(arr, cmap='viridis')
                plt.title(f'Technical Attention (Layer {layer_idx+1})')
                buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0)
                img = Image.open(buf).resize((500,500))
                self.writer.add_image(f'attention/technical_layer_{layer_idx+1}',
                                    np.array(img).transpose(2,0,1), step)
                plt.close(fig); buf.close()

            gc.collect()

        # ------------------------------------------------------------------ episode-PnL trend
        if self.episode_pnls:
            fig = plt.figure(figsize=(24, 8))
            plt.plot(self.episode_pnls, color='blue'); plt.axhline(0,color='r',ls='--',alpha=.3)
            plt.xlabel('Episode'); plt.ylabel('PnL %'); plt.title('PnL over Episodes'); plt.grid(alpha=.3)
            buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=120, bbox_inches='tight'); buf.seek(0)
            img = Image.open(buf).resize((1000,400))
            self.writer.add_image('performance/pnl_trend', np.array(img).transpose(2,0,1), self.step)
            plt.close(fig); buf.close()

        # optional scalar of latest reward
        if self.reward_buffer.head > 0 or self.reward_buffer.is_full:
            reward_array = self.reward_buffer.get_ordered()
            self.writer.add_scalar('training/reward', np.mean(reward_array), self.step)

        # ADDED: Forcibly log current Q-values if any exist
        if self.q_value_buffer['main'].head > 0 or self.q_value_buffer['main'].is_full:
            main_q_array = self.q_value_buffer['main'].get_ordered()
            q_mean = np.mean(main_q_array)
            self.writer.add_scalar('q_values/forced_main_mean', q_mean, self.step)
            logging.info(f"Force-logged main Q-value mean: {q_mean} at step {self.step}")
            self.q_values_logged = True
            
        if self.q_value_buffer['target'].head > 0 or self.q_value_buffer['target'].is_full:
            target_q_array = self.q_value_buffer['target'].get_ordered()
            q_mean = np.mean(target_q_array)
            self.writer.add_scalar('q_values/forced_target_mean', q_mean, self.step)
            logging.info(f"Force-logged target Q-value mean: {q_mean} at step {self.step}")
            self.q_values_logged = True

        # actually write to disk
        self.writer.flush()
        
        # Debug summary
        logging.info(f"Flush complete. Wrote {scalar_count} scalars and {histogram_count} histograms to TensorBoard.")

        # ------------------------------------------------------------------ clear buffers
        self.offline_scalars.clear()
        self.offline_histograms.clear()
        self.offline_attention.clear()
        self.offline_pnl.clear()
        self.episode_pnls.clear()
        self.offline_feature_importance.clear()
        self.offline_feature_importance_heatmap.clear()

    
    def close(self):
        """Cleanup the writer."""
        if hasattr(self, 'writer'):
            self.writer.close()