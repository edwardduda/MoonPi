import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from classes.Config import Config

class FeatureAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, num_features):
        feature_dim = 12
        super().__init__()
        self.num_features = num_features 
        self.feature_dim = feature_dim

        self.proj = nn.Linear(embed_dim, num_features * feature_dim)
        
        # Multihead attention that operates over the feature tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # We'll be working with (batch, tokens, feature_dim)
        )
        
        # A simple feed-forward network applied on each feature token.
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Layer norms for residual connections.
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Optionally, a final projection back to embed_dim.
        self.out_proj = nn.Linear(num_features * feature_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            out: Tensor of shape (batch, seq_len, embed_dim) after feature attention.
            attn_weights: The attention weights from the multihead attention (for inspection).
        """
        batch, seq_len, _ = x.size()
        
        # Project input: shape becomes (batch, seq_len, num_features * feature_dim)
        x_proj = self.proj(x)
        
        # Reshape to have a token for each feature:
        # New shape: (batch * seq_len, num_features, feature_dim)
        x_feat = x_proj.view(batch * seq_len, self.num_features, self.feature_dim)
        
        # Create a key padding mask.
        # For each token (i.e. each feature vector), if the absolute sum is nearly zero, mark it as padded.
        # This mask should have shape (batch * seq_len, num_features) and be of type bool.
        key_padding_mask = (x_feat.abs().sum(dim=-1) < 1e-6)
        
        # Pass the mask to the multihead attention module.
        # Tokens flagged in the key_padding_mask will be ignored in the attention computation.
        attn_output, attn_weights = self.attention(
            x_feat, x_feat, x_feat, key_padding_mask=key_padding_mask
        )
        
        # Residual connection + layer norm.
        x_feat = self.layer_norm1(x_feat + self.dropout(attn_output))
        
        # Feed-forward network on each feature token.
        ffn_output = self.ffn(x_feat)
        x_feat = self.layer_norm2(x_feat + self.dropout(ffn_output))
        
        # Reshape back to (batch, seq_len, num_features * feature_dim)
        out = x_feat.view(batch, seq_len, self.num_features * self.feature_dim)
        
        # Project back to the original embedding dimension.
        out = self.out_proj(out)
        
        return out, attn_weights

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
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
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
        print(f"Total features in state: {self.num_features}") 
        
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
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, action_dim)
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