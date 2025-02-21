import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from classes.Config import Config

class TechnicalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, tech_num_heads, dropout_rate, num_techs, tech_dim):
        super().__init__()
        self.tech_dim = tech_dim
        self.num_techs = num_techs
        
        self.proj = nn.Linear(embed_dim, self.num_techs * self.tech_dim)
        
        # Multihead attention that operates over the feature tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=self.tech_dim,
            num_heads=tech_num_heads,
            dropout=dropout_rate,
            batch_first=True  # We'll be working with (batch, tokens, tech_dim)
        )
        
        # A simple feed-forward network applied on each feature token.
        self.ffn = nn.Sequential(
            nn.Linear(self.tech_dim, self.tech_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.tech_dim * 4, self.tech_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.tech_dim * 2, self.tech_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(self.tech_dim)
        self.layer_norm2 = nn.LayerNorm(self.tech_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.out_proj = nn.Linear(self.num_techs * self.tech_dim, embed_dim)
        
    def forward(self, x):
        batch, seq_len, _ = x.size()
        
        # Project input: shape becomes (batch, seq_len, num_techs * tech_dim)
        x_proj = self.proj(x)
        
        # Reshape to have a token for each feature:
        # New shape: (batch * seq_len, num_techs, tech_dim)
        x_feat = x_proj.view(batch * seq_len, self.num_techs, self.tech_dim)

        # Create a key padding mask.
        key_padding_mask = (x_feat.abs().sum(dim=-1) < 1e-6)
        
        # Pass the mask to the multihead attention module.
        attn_output, attn_weights = self.attention(
            x_feat, x_feat, x_feat, key_padding_mask=key_padding_mask
        )
        
        # Residual connection + layer norm.
        x_feat = self.layer_norm1(x_feat + self.dropout(attn_output))
        
        # Feed-forward network on each feature token.
        ffn_output = self.ffn(x_feat)
        x_feat = self.layer_norm2(x_feat + self.dropout(ffn_output))
        
        # Reshape back to (batch, seq_len, num_techs * tech_dim)
        out = x_feat.view(batch, seq_len, self.num_techs * self.tech_dim)
        
        # Project back to the original embedding dimension.
        out = self.out_proj(out)
        
        return out, attn_weights

class AstroAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_astro_heads, dropout_rate, num_features, feature_dim):
        super().__init__()
        self.num_features = num_features 
        self.astro_feature_dim = feature_dim

        self.proj = nn.Linear(embed_dim, num_features * self.astro_feature_dim)
        
        # Multihead attention that operates over the feature tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=self.astro_feature_dim,
            num_heads=num_astro_heads,
            dropout=dropout_rate,
            batch_first=True  # We'll be working with (batch, tokens, feature_dim)
        )
        
        # A simple feed-forward network applied on each feature token.
        self.ffn = nn.Sequential(
            nn.Linear(self.astro_feature_dim, self.astro_feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.astro_feature_dim * 4, self.astro_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.astro_feature_dim * 2, self.astro_feature_dim)
        )
        
        # Layer norms for residual connections.
        self.layer_norm1 = nn.LayerNorm(self.astro_feature_dim)
        self.layer_norm2 = nn.LayerNorm(self.astro_feature_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.out_proj = nn.Linear(self.num_features * self.astro_feature_dim, embed_dim)
        
    def forward(self, x):
        batch, seq_len, _ = x.size()
        
        x_proj = self.proj(x)
        x_feat = x_proj.view(batch * seq_len, self.num_features, self.astro_feature_dim)
        
        key_padding_mask = (x_feat.abs().sum(dim=-1) < 1e-6)
        
        attn_output, attn_weights = self.attention(
            x_feat, x_feat, x_feat, key_padding_mask=key_padding_mask
        )
        
        x_feat = self.layer_norm1(x_feat + self.dropout(attn_output))
        ffn_output = (self.ffn(x_feat)).float()
        x_feat = self.layer_norm2(x_feat + self.dropout(ffn_output))
        
        out = x_feat.view(batch, seq_len, self.num_features * self.astro_feature_dim)
        out = self.out_proj(out)
        
        return out, attn_weights

class TemporalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_temporal_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_temporal_heads,
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
        normed_x = self.layer_norm1(x)
        attention_output, attention_weights = self.attention(normed_x, normed_x, normed_x)
        
        x = x + self.dropout(attention_output)
        normed_x = self.layer_norm1(x)
        ffn_output = self.ffn(normed_x)
        x = x + self.dropout(ffn_output)
        
        return x, attention_weights

class AttentionDQN(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size):
        super().__init__()
        self.config = Config()
        # Define the holding flag index (update this as needed).
        self.holding_flag_index = -4
        self.dropout = self.config.TRAINING_PARMS.get('DROPOUT_RATE')
        self.state_dim = state_dim  # (seq_len, num_features)
        self.seq_len, self.num_features = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.num_astro_heads = self.config.ARCHITECTURE_PARMS.get("NUM_ASTRO_HEADS")
        self.num_temporal_heads = self.config.ARCHITECTURE_PARMS.get("NUM_TEMPORAL_HEADS")
        self.num_tech_heads = self.config.ARCHITECTURE_PARMS.get("NUM_TECHNICAL_HEADS")
        self.astro_dim = self.config.ARCHITECTURE_PARMS.get('ASTRO_DIM')
        self.embed_dim = self.config.ARCHITECTURE_PARMS.get("EMBED_DIM")
        self.tech_dim = self.config.ARCHITECTURE_PARMS.get('TECH_DIM')
        self.num_temporal_blocks = self.config.ARCHITECTURE_PARMS.get("NUM_TEMPORAL_LAYERS")
        self.num_astro_blocks = self.config.ARCHITECTURE_PARMS.get("NUM_ASTRO_LAYERS")
        self.num_tech_blocks = self.config.ARCHITECTURE_PARMS.get("NUM_TECH_LAYERS")
        
        # Define the technical tokens:
        self.num_tech_total = 11  # originally expected technical tokens (including the holding flag)
        self.num_tech = self.num_tech_total - 1  # processed by technical blocks (holding flag removed)
        
        # Update input projection to expect one less feature (holding flag removed).
        self.input_projection = nn.Sequential(
            nn.Linear(self.num_features - 1, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding.
        self.register_buffer('pos_encoding', self._get_sinusoidal_encoding(self.seq_len, self.embed_dim))
        self.pos_dropout = nn.Dropout(self.dropout)
        
        # Technical attention blocks (update token count to self.num_tech).
        self.tech_blocks = nn.ModuleList([
            TechnicalAttentionBlock(
                self.embed_dim, self.num_tech_heads, self.dropout, self.num_tech, self.config.ARCHITECTURE_PARMS.get("TECH_DIM"))
            for _ in range(self.num_tech_blocks)
        ])
        
        # Temporal attention blocks.
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(self.embed_dim, self.num_temporal_heads, self.dropout)
            for _ in range(self.num_temporal_blocks)
        ])

        # Astro attention blocks operate on the remaining tokens.
        self.astro_blocks = nn.ModuleList([
            AstroAttentionBlock(self.embed_dim, self.num_astro_heads, self.dropout, self.num_features - self.num_tech_total, self.astro_dim)
            for _ in range(self.num_astro_blocks)
        ])

        self.final_norm = nn.LayerNorm(self.embed_dim)
        
        self.q_values = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 2, action_dim)
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
        # x shape: (batch, seq_len, num_features)
        batch, seq_len, num_features = x.shape
        if (seq_len, num_features) != self.state_dim:
            raise ValueError(f"Expected state dimensions {self.state_dim} but got ({seq_len}, {num_features})")
        
        # 1. Separate the holding flag.
        # The holding flag is assumed to be at self.holding_flag_index.
        holding_flag = x[:, :, self.holding_flag_index].unsqueeze(-1)  # shape: (batch, seq_len, 1)
        
        # 2. Remove the holding flag from the remaining features.
        x_proc = torch.cat([x[:, :, :self.holding_flag_index], x[:, :, self.holding_flag_index+1:]], dim=-1)
        # Now x_proc has shape (batch, seq_len, num_features - 1)
        
        # 3. Run the remaining features through the network.
        x_proc = self.input_projection(x_proc)
        x_proc = x_proc + self.pos_dropout(self.pos_encoding[:, :seq_len, :])
        
        # Create a mask for any padded tokens.
        zero_mask = (x_proc.abs().sum(dim=-1, keepdim=True) == 0)
        x_proc = x_proc.masked_fill(zero_mask, 0.0)
        
        # Technical attention blocks.
        technical_weights = []
        for block in self.tech_blocks:
            x_proc, weights = block(x_proc)
            technical_weights.append(weights)
            x_proc = x_proc.masked_fill(zero_mask, 0.0)
        
        # Temporal attention blocks.
        temporal_weights = []
        for block in self.temporal_blocks:
            x_proc, weights = block(x_proc)
            temporal_weights.append(weights)
            x_proc = x_proc.masked_fill(zero_mask, 0.0)
        
        # Astro attention blocks.
        feature_weights = []
        for block in self.astro_blocks:
            x_proc, weights = block(x_proc)
            feature_weights.append(weights)
            x_proc = x_proc.masked_fill(zero_mask, 0.0)
        
        # Global average pooling.
        valid_tokens = (~zero_mask).half()
        x_proc = (x_proc * valid_tokens).sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
        
        # Final normalization and projection.
        x_proc = self.final_norm(x_proc)
        q_values = self.q_values(x_proc)
        
        # 4. Gate the final Q-values with the holding flag.
        # Aggregate the holding flag over the sequence dimension and apply a sigmoid.
        gate = torch.sigmoid(holding_flag.mean(dim=1))  # shape: (batch, 1)
        q_values = q_values * gate
        
        return q_values, (technical_weights, temporal_weights, feature_weights)
