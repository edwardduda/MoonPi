import torch

class Config:
    def __init__(self):
        self.DATA_CONFIG = {
        'SEGMENT_SIZE' : int(100), #time window size      252 = 1 fiscal year
        'DATASET_CSV' : "full_df.csv",
        'NUM_FEATURES' : None
        }
        
        self.TRAINING_PARMS = {
        'EPISODES' : 5000,
        'BATCH_SIZE' : 16,
        'BUFFER_SIZE' : 120000,
        'MIN_REPLAY_SIZE' : 90000,
        'MIN_LEARNING_RATE' : 1e-5,
        'LEARNING_RATE' : 1e-4,
        'GAMMA' : 0.99,
        'TAU' : 0.07,
        'EMBED_DIM' : 384,
        'NUM_HEADS' : 12,
        'DROPOUT_RATE' : 0.1,
        'DEVICE' : "mps" if torch.mps.is_available() else 'cpu',
        'NUM_FEATURE_LAYERS' : 1,
        'NUM_TEMPORAL_LAYERS' : 1,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.25,
        'EPSILON_RESET' : 0.3,
        'EPSILON_DECAY': 0.99995,
        'WEIGHT_DECAY' : 1e-5,
        'STEPS_PER_EPISODE' : self.DATA_CONFIG.get('SEGMENT_SIZE'),
        'MAX_GRADIENT_CLIP' : 1.0
        }
        
        self.MARKET_ENV_PARMS = {
        'INITIAL_CAPITAL' : 1000.0,
        'SEGMENT_SIZE' : 179, #number of features
        'MAX_HOLD_STEPS' : 21,
        'HOLD_PENALTY' : 0.001,
        'TRADING_FEE' : 0.60,
        'MAX_TRADES_PER_MONTH' : 15,
        }