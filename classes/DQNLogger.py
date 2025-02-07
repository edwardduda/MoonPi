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

class DQNLogger:
    def __init__(self, log_dir, scalar_freq, attention_freq, histogram_freq, buffer_size):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f"{log_dir}/{timestamp}")
        self.step = 0
        self.scalar_freq = scalar_freq
        self.attention_freq = attention_freq
        self.histogram_freq = histogram_freq
        self.buffer_size = buffer_size
        self.reward_buffer = deque(maxlen=buffer_size)
        self.loss_buffer = deque(maxlen=buffer_size)
        self.pnl_history = deque(maxlen=100)
        self.q_value_buffer = {
            'main': deque(maxlen=buffer_size),
            'target': deque(maxlen=buffer_size)
        }
        
        # Initialize attention averaging buffers
        self.temporal_attention_buffer = None
        self.feature_attention_buffer = None
        self.attention_buffer_count = 0
        self.max_attention_samples = 1000  # Number of samples to average over
        
        
    def initialize_attention_buffers(self, temporal_weights, feature_weights):
        """Initialize attention buffers with the right shapes"""
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
    
    def update_attention_buffers(self, temporal_weights, feature_weights):
        """Update running average of attention weights"""
        with torch.no_grad():
            # Initialize buffers if needed
            self.initialize_attention_buffers(temporal_weights, feature_weights)
            
            # Update count for averaging
            self.attention_buffer_count = min(self.attention_buffer_count + 1, self.max_attention_samples)
            
            # Update running average for temporal attention
            for layer_idx, layer_weights in enumerate(temporal_weights):
                if isinstance(layer_weights, tuple):
                    layer_weights = layer_weights[0]
                    
                # Exponential moving average update
                alpha = 1.0 / self.attention_buffer_count
                self.temporal_attention_buffer[layer_idx] = (
                    (1 - alpha) * self.temporal_attention_buffer[layer_idx] +
                    alpha * layer_weights[0]  # First batch item
                )
            
            # Update running average for feature attention
            for layer_idx, layer_weights in enumerate(feature_weights):
                if isinstance(layer_weights, tuple):
                    layer_weights = layer_weights[0]
                    
                # Exponential moving average update
                self.feature_attention_buffer[layer_idx] = (
                    (1 - alpha) * self.feature_attention_buffer[layer_idx] +
                    alpha * layer_weights[0]  # First batch item
                )
    
    def log_attention_heatmaps(self, temporal_weights, feature_weights):
        """Log averaged attention weights as heatmaps for all layers"""
        plt.ioff()
        
        try:
            # Add shape debugging
            print("Temporal weights shapes:", [w.shape if isinstance(w, torch.Tensor) else type(w) for w in temporal_weights])
            print("Feature weights shapes:", [w.shape if isinstance(w, torch.Tensor) else type(w) for w in feature_weights])
            
            # Update running averages only if shapes are correct
            valid_weights = True
            for w in temporal_weights + feature_weights:
                if not isinstance(w, torch.Tensor):
                    valid_weights = False
                    break
                    
            if valid_weights:
                self.update_attention_buffers(temporal_weights, feature_weights)
                
                # Visualize temporal attention weights
                for layer_idx, avg_weights in enumerate(self.temporal_attention_buffer):
                    if not isinstance(avg_weights, torch.Tensor):
                        continue
                        
                    # Get averaged weights and ensure they're 2D
                    temp_weights = avg_weights.cpu().numpy()
                    if len(temp_weights.shape) > 2:
                        temp_weights = temp_weights.mean(axis=0)
                    elif len(temp_weights.shape) < 2:
                        continue
                    
                    fig = plt.figure(figsize=(40, 40))
                    sns.heatmap(temp_weights, cmap='viridis')
                    plt.title(f'Temporal Attention Pattern (Layer {layer_idx + 1})')
                    plt.xlabel('Time Steps')
                    plt.ylabel('Attention Position')
                    
                    # Save visualization
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150)
                    buf.seek(0)
                    image = Image.open(buf)
                    image = image.resize((2000, 2000))
                    self.writer.add_image(f'attention/temporal_layer_{layer_idx + 1}', 
                                        np.array(image).transpose(2, 0, 1), 
                                        self.step)
                    
                    plt.close(fig)
                    buf.close()
                
                # Visualize feature attention weights
                for layer_idx, avg_weights in enumerate(self.feature_attention_buffer):
                    if not isinstance(avg_weights, torch.Tensor):
                        continue
                        
                    # Get averaged weights and ensure they're 2D
                    feat_weights = avg_weights.cpu().numpy()
                    if len(feat_weights.shape) > 2:
                        feat_weights = feat_weights.mean(axis=0)
                    elif len(feat_weights.shape) < 2:
                        continue
                    
                    # Only visualize if dimensions match feature count
                    if feat_weights.shape[0] == len(self.feature_names):
                        fig = plt.figure(figsize=(40, 40))
                        sns.heatmap(feat_weights, cmap='viridis',
                                xticklabels=self.feature_names,
                                yticklabels=self.feature_names)
                        plt.title(f'Feature Attention Pattern (Layer {layer_idx + 1})')
                        plt.xticks(rotation=45, ha='right')
                        plt.yticks(rotation=0)
                        
                        # Save visualization
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        image = Image.open(buf)
                        image = image.resize((2000, 2000))
                        self.writer.add_image(f'attention/feature_layer_{layer_idx + 1}', 
                                            np.array(image).transpose(2, 0, 1), 
                                            self.step)
                        
                        plt.close('all')
                        buf.close()
                    else:
                        print(f"Feature weight dimensions don't match feature count. "
                            f"Got shape {feat_weights.shape}, expected first dim to be {len(self.feature_names)}")
                    
                    gc.collect()
            
        except Exception as e:
            print(f"Warning: Error in attention visualization: {e}")
            print("Temporal weights info:", [type(w) for w in temporal_weights])
            print("Feature weights info:", [type(w) for w in feature_weights])
        
        finally:
            plt.ion()
        
    def log_metrics(self, metrics):
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(metric_name, value, self.step)
            elif torch.is_tensor(value) and value.numel() == 1:
                self.writer.add_scalar(metric_name, value.item(), self.step)
                      
    def log_feature_importance(self, feature_weights, feature_names):
        """
        Log averaged attention weights for each feature with improved safety checks
        """
        plt.ioff()
        try:
            # Validate inputs
            if not feature_weights or not feature_names:
                return
            
            # Check dimensions
            for layer_idx, layer_weights in enumerate(feature_weights):
                with torch.no_grad():
                    # Convert to numpy and take mean across attention positions
                    avg_weights = layer_weights.detach().mean(dim=0).cpu().numpy()
                
                    # Ensure dimensions match
                    feature_dim = avg_weights.shape[-1]
                    if feature_dim != len(feature_names):
                        logging.warning(f"Feature dimension mismatch: weights dim {feature_dim} != feature_names dim {len(feature_names)}")
                        continue
                    
                    # Calculate feature importance
                    feature_importance = avg_weights.mean(axis=0)
                
                    # Sort features by importance and get top N (where N is min of 30 or available features)
                    n_features = min(30, len(feature_names))
                    sorted_idx = np.argsort(feature_importance)[-n_features:]
                    pos = np.arange(len(sorted_idx))
                
                    # Create visualization
                    fig = plt.figure(figsize=(40, 40))
                    plt.barh(pos, feature_importance[sorted_idx], height=0.8)
                    plt.yticks(pos, np.array(feature_names)[sorted_idx], fontsize=18)
                    plt.xlabel('Average Attention Weight', fontsize=20)
                    plt.title(f'Most Important Features (Layer {layer_idx + 1})', fontsize=18, pad=20)
                    plt.grid(True, axis='x', alpha=0.3)
                    plt.tight_layout()
                
                    # Save and cleanup
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')
                    buf.seek(0)
                    image = Image.open(buf)
                    image = image.resize((1500, 1500))
                    self.writer.add_image(f'feature_importance/layer_{layer_idx + 1}', 
                                    np.array(image).transpose(2, 0, 1), 
                                    self.step)
                    plt.close(fig)
                    buf.close()
                
        except Exception as e:
            logging.error(f"Error in feature importance visualization: {e}")
        
        finally:
            plt.ion()

    def log_training_step(self, epsilon, lr, reward, loss, main_q_values, target_q_values, 
                        temporal_weights=None, feature_weights=None):
        """Log training metrics with improved error handling."""
        try:
            # Safely accumulate metrics
            if reward is not None:
                self.reward_buffer.append(reward)
            if loss is not None:
                self.loss_buffer.append(loss)
            
            # Safely store Q-values
            with torch.no_grad():
                if main_q_values is not None:
                    self.q_value_buffer['main'].append(main_q_values.mean().item())
                if target_q_values is not None:
                    self.q_value_buffer['target'].append(target_q_values.mean().item())
        
            # Log scalars at specified frequency
            if self.step % self.scalar_freq == 0 and len(self.reward_buffer) > 0 and len(self.loss_buffer) > 0:
                metrics = {
                'training/reward': np.mean(self.reward_buffer),
                'training/loss': np.mean(self.loss_buffer),
                'training/reward_std': np.std(self.reward_buffer) if len(self.reward_buffer) > 1 else 0,
                'training/loss_std': np.std(self.loss_buffer) if len(self.loss_buffer) > 1 else 0,
                }
            
                if len(self.q_value_buffer['main']) > 0:
                    metrics['q_values/main_mean'] = np.mean(self.q_value_buffer['main'])
                if len(self.q_value_buffer['target']) > 0:
                    metrics['q_values/target_mean'] = np.mean(self.q_value_buffer['target'])
                
                self.log_metrics(metrics)
        
            # Log histograms at specified frequency
            if self.step % self.histogram_freq == 0:
                with torch.no_grad():
                    if main_q_values is not None:
                        self.writer.add_histogram('q_values/main_dist', 
                                                main_q_values.cpu(), 
                                                self.step)
                    if target_q_values is not None:
                        self.writer.add_histogram('q_values/target_dist', 
                                                target_q_values.cpu(), 
                                                self.step)
        
            # Log attention heatmaps at specified frequency
            if (temporal_weights is not None and 
                feature_weights is not None and 
                self.step % self.attention_freq == 0):
                self.log_attention_heatmaps(temporal_weights, feature_weights)
            
            self.step += 1
        
        except Exception as e:
            logging.error(f"Error in log_training_step: {e}")
            
    def log_episode_pnl(self, initial_capital, final_portfolio_value):
        """
        Log the cumulative PnL percentage for an episode with visualization
        """
        plt.ioff()  # Turn off interactive mode
        try:
            pnl_percentage = ((final_portfolio_value - initial_capital) / initial_capital) * 100
            self.pnl_history.append(pnl_percentage)
        
            # Log to tensorboard
            self.writer.add_scalar('performance/episode_pnl_percentage', pnl_percentage, self.step)
        
            # Create PnL trend plot
            fig = plt.figure(figsize=(15, 9))
            plt.plot(list(self.pnl_history), color='blue', label='PnL %')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title('Trading PnL Performance Over Episodes')
            plt.xlabel('Episodes')
            plt.ylabel('PnL %')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
            # Save plot to tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = image.resize((800, 600))
            self.writer.add_image('performance/pnl_trend', 
                                np.array(image).transpose(2, 0, 1), 
                                self.step)
        
            plt.close(fig)
            buf.close()
        
            print(f"\nEpisode PnL: {pnl_percentage:.2f}% | Avg PnL (last 100): {np.mean(self.pnl_history):.2f}%")
        
            return pnl_percentage
        
        except Exception as e:
            print(f"Error in PnL visualization: {e}")
            return 0
        finally:
            plt.ion()  
    
    def log_feature_importance_heatmap(self, feature_weights, feature_names):
        """
        Create a heatmap of top feature importances across layers
        """
        plt.ioff()
        try:
            n_layers = len(feature_weights)
            with torch.no_grad():
                # Get importance scores for each layer
                layer_importances = []
                for layer_weights in feature_weights:
                    avg_weights = layer_weights.detach().mean(dim=0).cpu().numpy()
                    importance = avg_weights.mean(axis=0)
                    layer_importances.append(importance)
            
                # Convert to numpy array
                importance_matrix = np.array(layer_importances)
            
                # Get top 20 features by average importance across layers
                avg_importance = importance_matrix.mean(axis=0)
                top_indices = np.argsort(avg_importance)[-20:]
            
                # Create heatmap
                fig = plt.figure(figsize=(40, 40))
                sns.heatmap(importance_matrix[:, top_indices].T, 
                        xticklabels=[f'Layer {i+1}' for i in range(n_layers)],
                        yticklabels=np.array(feature_names)[top_indices],
                        cmap='viridis', annot=True, fmt='.4f',
                        cbar_kws={'label': 'Importance Score'})
            
                plt.title('Feature Importance Across Layers (Top 20 Features)')
                plt.tight_layout()
                plt.xticks(rotation=90, ha='right', fontsize=12)
                plt.yticks(rotation=0, fontsize=12)
                # Save plot
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=160, bbox_inches='tight')
                buf.seek(0)
                image = Image.open(buf)
                image = image.resize((2200, 2200))
                self.writer.add_image('feature_importance/layer_comparison', 
                                np.array(image).transpose(2, 0, 1), 
                                self.step)
            
                plt.close(fig)
                buf.close()
            
        except Exception as e:
            print(f"Warning: Error in feature importance heatmap visualization: {e}")
    
        finally:
            plt.ion()
            
    def close(self):
        """Cleanup method for the logger"""
        if hasattr(self, 'writer'):
            self.writer.close()