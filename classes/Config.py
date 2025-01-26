import torch

class Config:
    def __init__(self):
        self.DATA_CONFIG = {
        'SEGMENT_SIZE' : int(280), #time window size      252 = 1 fiscal year
        'DATASET_CSV' : "full_df.csv",
        'NUM_FEATURES' : None
        }
        
        self.TRAINING_PARMS = {
        'EPISODES' : 5000,
        'BATCH_SIZE' : 128,
        'BUFFER_SIZE' : 150000,
        'MIN_REPLAY_SIZE' : 110000,
        'LEARNING_RATE' : 1e-4,
        'MIN_LR' : 1e-5,
        'GAMMA' : 0.994,
        'TAU' : 0.06,
        'EMBED_DIM' : 392,
        'NUM_HEADS' : 16,
        'DROPOUT_RATE' : 0.1,
        'DEVICE' : "mps" if torch.mps.is_available() else 'cpu',
        'NUM_TEMPORAL_LAYERS' : 2,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.3,
        'EPSILON_DECAY': 0.999995,
        'WEIGHT_DECAY' : 1e-5,
        'STEPS_PER_EPISODE' : self.DATA_CONFIG.get('SEGMENT_SIZE'),
        'MAX_GRADIENT_CLIP' : 1.0
        }
        
        self.MARKET_ENV_PARMS = {
        'INITIAL_CAPITAL' : 1000.0,
        'SEGMENT_SIZE' : 179, #number of features
        'MAX_HOLD_STEPS' : 23,
        'HOLD_PENALTY' : 0.001,
        'TRADING_FEE' : 0.50,
        'MAX_TRADES_PER_MONTH' : 10,
        }
        

        
        