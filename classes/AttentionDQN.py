import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from classes.Config import Config


class FeatureAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, num_features):
        super().__init__()
        config = Config()
        self.num_features = num_features
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm([embed_dim, config.DATA_CONFIG.get('SEGMENT_SIZE')])
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # Add debug prints
        print(f"Input shape: {x.shape}")
        
        # Transpose to (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)
        print(f"After first transpose shape: {x.shape}")
        
        # Layer norm
        normed_x = self.layer_norm1(x)
        # Transpose back for attention
        normed_x = normed_x.transpose(1, 2)
        print(f"Before attention shape: {normed_x.shape}")
        
        # Apply attention
        attention_output, attention_weights = self.attention(normed_x, normed_x, normed_x)
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Process attention weights to get feature attention
        # If attention_weights is 2D, reshape it appropriately
        if len(attention_weights.shape) == 2:
            # Reshape to (batch_size, seq_len, seq_len)
            seq_len = attention_weights.size(1)
            attention_weights = attention_weights.view(-1, seq_len, seq_len)
        
        # Take only the feature-relevant portion
        feature_attention = attention_weights[:, :self.num_features, :self.num_features]
        print(f"Feature attention shape: {feature_attention.shape}")
        
        # Continue with the rest of the forward pass
        attention_output = attention_output.transpose(1, 2)
        x = x + self.dropout(attention_output)
        
        normed_x = self.layer_norm1(x)
        
        batch_size, embed_dim, seq_len = normed_x.shape
        reshaped_x = normed_x.transpose(1, 2).reshape(-1, embed_dim)
        ffn_output = self.ffn(reshaped_x)
        ffn_output = ffn_output.view(batch_size, seq_len, embed_dim).transpose(1, 2)
        
        x = x + self.dropout(ffn_output)
        x = x.transpose(1, 2)
        
        return x, feature_attention

class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # Add debug prints
        #print(f"Temporal input shape: {x.shape}")
        
        # x shape: (batch_size, seq_len, embed_dim)
        normed_x = self.layer_norm1(x)
        attention_output, attention_weights = self.attention(normed_x, normed_x, normed_x)
        #print(f"Temporal attention weights shape: {attention_weights.shape}")
        
        x = x + self.dropout(attention_output)
        
        normed_x = self.layer_norm1(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.dropout(ffn_output)
        
        return x, attention_weights

class AttentionDQN(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, num_heads, dropout_rate, batch_size):
        super().__init__()
        self.state_dim = state_dim
        self.seq_len, self.num_features = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        print(f"Initializing AttentionDQN with dimensions: seq_len={self.seq_len}, num_features={self.num_features}")
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._get_sinusoidal_encoding(self.seq_len, embed_dim))
        self.pos_dropout = nn.Dropout(dropout_rate)
        
        # Temporal blocks remain the same
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(embed_dim, num_heads, dropout_rate)
            for _ in range(1)
        ])
        
        # Feature blocks now get num_features parameter
        self.feature_blocks = nn.ModuleList([
            FeatureAttentionBlock(embed_dim, num_heads, dropout_rate, self.num_features)
            for _ in range(1)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        self.q_values = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, action_dim)
        )
        
        self._init_weights()
    
    def _get_sinusoidal_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(1, seq_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        if (seq_len, num_features) != self.state_dim:
            raise ValueError(f"Expected state dimensions {self.state_dim} but got ({seq_len}, {num_features})")
        
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_dropout(self.pos_encoding[:, :seq_len, :])
        
        # Create attention mask for padding
        zero_mask = (x.abs().sum(dim=-1, keepdim=True) == 0)
        x = x.masked_fill(zero_mask, 0.0)
        
        # Apply temporal attention
        temporal_weights = []
        for block in self.temporal_blocks:
            x, weights = block(x)
            temporal_weights.append(weights)
            x = x.masked_fill(zero_mask, 0.0)
        
        # Apply feature attention
        feature_weights = []
        for block in self.feature_blocks:
            x, weights = block(x)
            feature_weights.append(weights)
            x = x.masked_fill(zero_mask, 0.0)
        
        # Global average pooling
        valid_tokens = (~zero_mask).float()
        x = (x * valid_tokens).sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
        
        # Final processing
        x = self.final_norm(x)
        q_values = self.q_values(x)
        
        return q_values, (temporal_weights, feature_weights)