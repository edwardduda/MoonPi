import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from classes.Config import Config
import cProfile
import pstats

class AttentionChunk(nn.Module):
    """
    One TECH → ASTRO → TEMP pipeline with fresh weights.
    """
    def __init__(
        self,
        embed_dim,
        num_tech_heads, tech_dim, num_techs,
        num_astro_heads, astro_dim, num_nontech_feats,
        num_temporal_heads,
        dropout,
        block_type,
        multiplier
    ):
        super().__init__()
        self.block_type = block_type
        self.multiplier = multiplier
        
        self.tech  = TechnicalAttentionBlock(
            embed_dim, num_tech_heads, dropout, num_techs, tech_dim
        )
        self.astro = AstroAttentionBlock(
            embed_dim, num_astro_heads, dropout, num_nontech_feats, astro_dim
        )
        self.temp  = TemporalAttentionBlock(
            embed_dim, num_temporal_heads, dropout
        )
                
    def forward(self, x, zero_mask, w_collector):
        # TECH
        if self.block_type == "tech":
            x, w = self.tech(x);  x = x.masked_fill(zero_mask, 0)
            w_collector["tech"].append(w)
            
            x, w = self.tech(x);  x = x.masked_fill(zero_mask, 0)
            w_collector["tech"].append(w)
            
            x, w = self.tech(x);  x = x.masked_fill(zero_mask, 0)
            w_collector["tech"].append(w)
            
            x, w = self.tech(x);  x = x.masked_fill(zero_mask, 0)
            w_collector["tech"].append(w)
            return x
        # ASTRO
        if self.block_type == "astro":
            x, w = self.astro(x); x = x.masked_fill(zero_mask, 0)
            w_collector["astro"].append(w)
            
            x, w = self.astro(x); x = x.masked_fill(zero_mask, 0)
            w_collector["astro"].append(w)
            
            return x
        # TEMP
        if self.block_type == "temp":
            x, w = self.temp(x);  x = x.masked_fill(zero_mask, 0)
            w_collector["temp"].append(w)

            return x
    
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
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.tech_dim * 4, self.tech_dim * 2),
            nn.SiLU(),
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
        ffn_output = self.ffn(x_feat).half().float()
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
        
        self.ffn = nn.Sequential(
            nn.Linear(self.astro_feature_dim, self.astro_feature_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.astro_feature_dim * 4, self.astro_feature_dim * 2),
            nn.SiLU(),
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
        ffn_output = self.ffn(x_feat).half().float()
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
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        normed_x = self.layer_norm1(x)
        attention_output, attention_weights = self.attention(normed_x, normed_x, normed_x)
        
        x = x + self.dropout(attention_output)
        normed_x = self.layer_norm1(x)
        ffn_output = self.ffn(normed_x).half().float()
        x = x + self.dropout(ffn_output)
        
        return x, attention_weights

class AttentionDQN(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size, num_chunks=2):
        super().__init__()
        self.cfg = Config()
        self.dropout = self.cfg.TRAINING_PARMS["DROPOUT_RATE"]
        self.seq_len, self.num_feats = state_dim
        self.action_dim = action_dim
        self.holding_flag_idx = -4          # same as before

        # ---- hyperparams from config ---- #
        hp = self.cfg.ARCHITECTURE_PARMS
        self.embed_dim      = hp["EMBED_DIM"]
        self.tech_dim       = hp["TECH_DIM"]
        self.astro_dim      = hp["ASTRO_DIM"]
        self.h_tech         = hp["NUM_TECHNICAL_HEADS"]
        self.h_astro        = hp["NUM_ASTRO_HEADS"]
        self.h_temp         = hp["NUM_TEMPORAL_HEADS"]
        self.n_tech         = 11                                # number of tech tokens
        self.n_nontech      = self.num_feats - self.n_tech       # astro feats

        # ---- input projection + PE ---- #
        self.input_proj = nn.Sequential(
            nn.Linear(self.num_feats - 1, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        self.register_buffer(
            "pos_enc", self._get_sinusoidal_encoding(self.seq_len, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(self.dropout)

        chunks = ["tech", "astro", "temp"]
        # ---- CHUNKS ---- #
        self.chunks = nn.ModuleList([
            AttentionChunk(
                self.embed_dim,
                self.h_tech,  self.tech_dim,  self.n_tech,
                self.h_astro, self.astro_dim, self.n_nontech,
                self.h_temp,
                self.dropout,
                chunk,
                multiplier=4
            ) for chunk in chunks
        ])
        # ---- head ---- #
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.q_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*4),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim*4, self.embed_dim*2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim*2, self.embed_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, action_dim)
        )
        self._init_weights()
        
        print(f"[INFO] Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
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
        B, L, F = x.shape
        assert (L, F) == (self.seq_len, self.num_feats)

        # split out holding flag & project
        hold_flag = x[:, :, self.holding_flag_idx].unsqueeze(-1)
        x = torch.cat([x[:, :, :self.holding_flag_idx],
                       x[:, :, self.holding_flag_idx+1:]], dim=-1)
        x = self.input_proj(x) + self.pos_drop(self.pos_enc[:, :L, :])

        zero_mask = (x.abs().sum(-1, keepdim=True) == 0)

        # collect attn weights
        w_collector = {"tech": [], "astro": [], "temp": []}

        # run through chunks
        for chunk in self.chunks:
            x = chunk(x, zero_mask, w_collector)
            
        # pool → head → q-values
        valid = ~zero_mask
        x = (x * valid).sum(1) / (valid.sum(1) + 1e-8)
        x = self.final_norm(x)
        q = self.q_head(x).half().float()

        # gate by holding flag (unchanged)
        gate = torch.sigmoid(hold_flag.mean(1))
        q = q * gate
        
        tech_weights_all = w_collector["tech"]
        astro_weights_all = w_collector["astro"]
        temp_weights_all = w_collector["temp"]
        
        return q, (tech_weights_all, temp_weights_all, astro_weights_all)



