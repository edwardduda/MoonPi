import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import datetime
from collections import deque
import gc
import logging
import scipy
from numba import jit  # imported in case you want to jit some helper functions later

jit(nopython=True, cache=True)
def running_mean_std(vals):
    acc, acc2 = 0.0, 0.0
    n = len(vals)
    for v in vals:
        acc  += v
        acc2 += v * v
    mean = acc / n
    std  = (acc2 / n - mean ** 2) ** 0.5
    return mean, std

class DQNLogger:
    def __init__(self, log_dir, scalar_freq, attention_freq, histogram_freq, buffer_size):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f"{log_dir}/{timestamp}")
        self.step = 0
        self.scalar_freq = scalar_freq
        self.attention_freq = attention_freq
        self.histogram_freq = histogram_freq
        self.buffer_size = buffer_size

        # Buffers for online averaging
        self.reward_buffer = deque(maxlen=buffer_size)
        self.loss_buffer = deque(maxlen=buffer_size)
        self.pnl_history = deque(maxlen=100)
        self.q_value_buffer = {
            'main': deque(maxlen=buffer_size),
            'target': deque(maxlen=buffer_size)
        }
        
        # Attention averaging buffers (updated each time, but plotting is delayed)
        self.temporal_attention_buffer = None
        self.feature_attention_buffer = None
        self.technical_attention_buffer = None
        self.attention_buffer_count = 0
        self.max_attention_samples = 1000  # samples to average over
        
        # Offline log buffers
        self.offline_scalars = {}      # metric_name -> list of (step, value)
        self.offline_histograms = {}   # metric_name -> list of (step, data)
        self.offline_attention = []    # list of dicts: { 'step', 'temporal', 'feature' }
        self.offline_pnl = []          # list of (step, pnl_percentage)
        self.episode_pnls = []         # list of all episode PnL percentages
        self.offline_feature_importance = []         # list of dicts: { 'step', 'layer', 'feature_importance', 'feature_names' }
        self.offline_feature_importance_heatmap = []   # list of dicts: { 'step', 'feature_weights', 'feature_names' }
        
        # Optionally, store feature names (used for attention heatmaps)
        self.feature_names = None

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
        """
        try:
            if reward is not None:
                self.reward_buffer.append(reward)
            if loss is not None:
                self.loss_buffer.append(loss)
            
            with torch.no_grad():
                if main_q_values is not None:
                    self.q_value_buffer['main'].append(main_q_values.mean().item())
                if target_q_values is not None:
                    self.q_value_buffer['target'].append(target_q_values.mean().item())
            
            if self.step % self.scalar_freq == 0 and self.reward_buffer and self.loss_buffer:
                r_mean, r_std = running_mean_std(np.array(self.reward_buffer, dtype=np.float32))
                metrics = {
                    'training/reward': np.mean(self.reward_buffer),
                    'training/loss': np.mean(self.loss_buffer),
                    'training/reward_std': r_mean if len(self.reward_buffer) > 1 else 0,
                    'training/loss_std': r_std if len(self.loss_buffer) > 1 else 0,
                    'training/epsilon': epsilon,
                    'training/lr' : lr
                }
                if self.q_value_buffer['main']:
                    metrics['q_values/main_mean'] = np.mean(self.q_value_buffer['main'])
                if self.q_value_buffer['target']:
                    metrics['q_values/target_mean'] = np.mean(self.q_value_buffer['target'])
                self.log_metrics(metrics)
            
            if self.step % self.histogram_freq == 0:
                with torch.no_grad():
                    if main_q_values is not None:
                        data = main_q_values.cpu().numpy()
                        self.offline_histograms.setdefault('q_values/main_dist', []).append((self.step, data))
                    if target_q_values is not None:
                        data = target_q_values.cpu().numpy()
                        self.offline_histograms.setdefault('q_values/target_dist', []).append((self.step, data))
            
            # Log attention data (including technical weights if provided)
            if (temporal_weights is not None and feature_weights is not None and 
                self.step % self.attention_freq == 0):
                self.log_attention_heatmaps(temporal_weights, feature_weights, technical_weights)
            
            self.step += 1
        
        except Exception as e:
            logging.error(f"Error in log_training_step: {e}")
    
    def log_episode_pnl(self, initial_capital, final_portfolio_value):
        """
        Store episode PnL data offline. The trend plot is compiled later.
        """
        plt.ioff()
        try:
            pnl_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
            self.pnl_history.append(pnl_percentage)
            self.episode_pnls.append(pnl_percentage)
            self.offline_scalars.setdefault('performance/episode_pnl_percentage', []).append((self.step, pnl_percentage))
            self.offline_pnl.append((self.step, pnl_percentage))
            
            print(f"\nEpisode PnL: {pnl_percentage:.2f}% | Avg PnL (last 100): {np.mean(self.pnl_history):.2f}%")
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
        # Flush scalars
        for metric, entries in self.offline_scalars.items():
            for step, value in entries:
                self.writer.add_scalar(metric, value, step)
        
        # Flush histograms
        for metric, entries in self.offline_histograms.items():
            for step, data in entries:
                self.writer.add_histogram(metric, data, step)
        
        # Flush attention logs (generate heatmaps)
        for log in self.offline_attention:
            step = log['step']
            # Temporal attention heatmaps
            for layer_idx, avg_weights in enumerate(log['temporal']):
                temp_weights = avg_weights.cpu().numpy()
                if len(temp_weights.shape) > 2:
                    temp_weights = temp_weights.mean(axis=0)
                elif len(temp_weights.shape) < 2:
                    continue
                fig = plt.figure(figsize=(36, 24))
                sns.heatmap(temp_weights, cmap='viridis')
                plt.title(f'Temporal Attention Pattern (Layer {layer_idx + 1})')
                plt.xlabel('Time Steps')
                plt.ylabel('Attention Position')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=120)
                buf.seek(0)
                image = Image.open(buf)
                image = image.resize((1000, 1000))
                self.writer.add_image(f'attention/temporal_layer_{layer_idx + 1}', 
                                      np.array(image).transpose(2, 0, 1), 
                                      step)
                plt.close(fig)
                buf.close()
            # Feature attention heatmaps
            for layer_idx, avg_weights in enumerate(log['feature']):
                feat_weights = avg_weights.cpu().numpy()
                if len(feat_weights.shape) > 2:
                    feat_weights = feat_weights.mean(axis=0)
                elif len(feat_weights.shape) < 2:
                    continue
                if self.feature_names and feat_weights.shape[0] == len(self.feature_names) - 11:
                    fig = plt.figure(figsize=(36, 24))
                    sns.heatmap(feat_weights, cmap='viridis',
                                xticklabels=self.feature_names,
                                yticklabels=self.feature_names)
                    plt.title(f'Feature Attention Pattern (Layer {layer_idx + 1})')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    image = Image.open(buf)
                    image = image.resize((1000, 1000))
                    self.writer.add_image(f'attention/feature_layer_{layer_idx + 1}', 
                                          np.array(image).transpose(2, 0, 1), 
                                          step)
                    plt.close(fig)
                    buf.close()
            # Technical attention heatmaps
            for layer_idx, avg_weights in enumerate(log.get('technical', [])):
                tech_weights = avg_weights.cpu().numpy()
                if len(tech_weights.shape) > 2:
                    tech_weights = tech_weights.mean(axis=0)
                elif len(tech_weights.shape) < 2:
                    continue
                fig = plt.figure(figsize=(24, 36))
                sns.heatmap(tech_weights, cmap='viridis')
                plt.title(f'Technical Attention Pattern (Layer {layer_idx + 1})')
                plt.xlabel('Token/Feature Index')
                plt.ylabel('Attention Position')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=200)
                buf.seek(0)
                image = Image.open(buf)
                image = image.resize((800, 800))
                self.writer.add_image(f'attention/technical_layer_{layer_idx + 1}', 
                                      np.array(image).transpose(2, 0, 1), 
                                      step)
                plt.close(fig)
                buf.close()
            gc.collect()
        
        # Flush episode PnL trend plot
        if self.episode_pnls:
            fig = plt.figure(figsize=(24, 36))
            plt.plot(self.episode_pnls, color='blue', label='PnL %')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title('Trading PnL Performance Over Episodes')
            plt.xlabel('Episodes')
            plt.ylabel('PnL %')
            plt.grid(True, alpha=0.3)
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = image.resize((800, 800))
            self.writer.add_image('performance/pnl_trend', 
                                np.array(image).transpose(2, 0, 1), 
                                self.step)
            plt.close(fig)
            buf.close()

        for entry in self.offline_feature_importance_heatmap:
            step = entry['step']
            feature_weights = entry['feature_weights']
            feature_names = entry['feature_names']
            n_layers = len(feature_weights)
            layer_importances = []
            for weights in feature_weights:
                avg_weights = weights.detach().mean(dim=0).cpu().numpy()
                importance = avg_weights.mean(axis=0)
                layer_importances.append(importance)
            importance_matrix = np.array(layer_importances)
            avg_importance = importance_matrix.mean(axis=0)
            top_indices = np.argsort(avg_importance)[-20:]
            fig = plt.figure(figsize=(36, 24))
            sns.heatmap(importance_matrix[:, top_indices].T, 
                        xticklabels=[f'Layer {i+1}' for i in range(n_layers)],
                        yticklabels=np.array(feature_names)[top_indices],
                        cmap='viridis', annot=True, fmt='.4f',
                        cbar_kws={'label': 'Importance Score'})
            plt.title('Feature Importance Across Layers (Top 20 Features)')
            plt.tight_layout()
            plt.xticks(rotation=90, ha='right', fontsize=12)
            plt.yticks(rotation=0, fontsize=12)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=160, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = image.resize((1500, 1500))
            self.writer.add_image('feature_importance/layer_comparison', 
                                np.array(image).transpose(2, 0, 1), 
                                step)
            plt.close(fig)
            buf.close()
        
        # Optionally flush the reward scalar if there's data in the reward buffer
        if self.reward_buffer:
            last_reward = np.mean(self.reward_buffer)
            self.writer.add_scalar('training/reward', last_reward, self.step)
        
        # Finally, flush the writer
        self.writer.flush()
        
        # Clear all offline buffers to avoid re-logging the same data
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
