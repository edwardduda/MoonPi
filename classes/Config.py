import torch
import MarketEnv

class Config:
    def __init__(self):
        self.DATA_CONFIG = {
            'SEGMENT_SIZE' : int(100), #time window size      252 = 1 fiscal year
            'DATASET_CSV' : "full_df.csv",
            'NUM_FEATURES' : None
        }
        
        self.TRAINING_PARMS = {
            'EPISODES' : 8000,
            'BATCH_SIZE' : 24,
            'BUFFER_SIZE' : 200000,
            'MIN_REPLAY_SIZE' : 180000,
            'MIN_LEARNING_RATE' : 1e-5,
            'LEARNING_RATE' : 1e-4,
            'GAMMA' : 0.9997,
            'TAU' : 0.06,
            'DROPOUT_RATE' : 0.1,
            'DEVICE' : "mps" if torch.mps.is_available() else 'cpu',
            'EPSILON_START': 1.0,
            'EPSILON_END': 0.25,
            'EPSILON_RESET' : 0.4,
            'EPSILON_DECAY': 0.99995,
            'WEIGHT_DECAY' : 1e-5,
            'STEPS_PER_EPISODE' : self.DATA_CONFIG.get('SEGMENT_SIZE'),
            'MAX_GRADIENT_CLIP' : 1.0
        }
        
        self.ARCHITECTURE_PARMS = {
            'ASTRO_DIM' : 24,
            'TECH_DIM' : 16,
            'NUM_ASTRO_LAYERS' : 1,
            'NUM_TEMPORAL_LAYERS' : 1,
            'NUM_TECH_LAYERS' : 1,
            'EMBED_DIM' : 384,
            'NUM_ASTRO_HEADS' : 12,
            'NUM_TEMPORAL_HEADS' : 6,
            'NUM_TECHNICAL_HEADS' : 4
        }
        
        self.MARKET_ENV_PARMS = {
        'NUM_PROJECTED_DAYS' : 15,
        'INITIAL_CAPITAL' : 1000.0,
        'SEGMENT_SIZE' : 179, #number of features
        'MAX_HOLD_STEPS' : 28,
        'HOLD_PENALTY' : 0.001,
        'TRADING_FEE' : 0.05,
        'MAX_TRADES_PER_MONTH' : 18,
        }