import torch
import pandas as pd
from scipy.stats import norm
from collections import namedtuple
import copy
import signal
import sys
from classes.MarketEnv import MarketEnv
from classes.AttentionDQN import AttentionDQN
from classes.Config import Config
from classes.Training import Training

# Global flag for graceful shutdown
should_exit = False

def check_state_dimensions(env, model, device):
    # Reset the environment to get an initial state.
    state = env.reset()
    print("Environment state shape:", state.shape)
    
    # Expected dimensions: (segment_size, combined_feature_dim)
    # Here, env.state_dim should equal the combined feature dimension.
    expected_shape = (env.segment_size, env.state_dim)
    print("Expected state shape:", expected_shape)
    
    if state.shape != expected_shape:
        print(f"Dimension mismatch! Got {state.shape} but expected {expected_shape}.")
    else:
        print("State dimensions are as expected.")

    # Convert state to a batch (with batch_size = 1) and send it to the specified device.
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # shape: (1, segment_size, combined_feature_dim)
    print("State tensor shape (after unsqueeze and to device):", state_tensor.shape)
    
    # Perform a forward pass through the model.
    q_values, attn = model(state_tensor)
    print("Model output Q-values shape:", q_values.shape)
    # If q_values shape is (1, action_dim) then everything looks good.
    
def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    global should_exit
    print('\nReceived signal to exit. Cleaning up...')
    should_exit = True
    
def initialize_models(state_dim, action_dim, config):
    print('made')
    main_model = AttentionDQN(
        state_dim= state_dim,
        action_dim= action_dim,
        batch_size=config.TRAINING_PARMS.get("BATCH_SIZE"),
        ).to(config.TRAINING_PARMS.get("DEVICE"))
    print('successful made')
    target_model = copy.deepcopy(main_model).to(config.TRAINING_PARMS.get("DEVICE"))
    target_model.eval() 

    return main_model, target_model

def cleanup(training_module=None, model=None):
    """Cleanup function to release resources"""
    try:
        if training_module and hasattr(training_module, 'logger'):
            training_module.logger.close()
        
        if model:
            # Save checkpoint
            print("\nSaving checkpoint...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_step': training_module.total_steps if training_module else 0,
                'checkpoint_type': 'interrupt'
            }, 'model_checkpoint_interrupt.pth')
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("Cleanup complete. Exiting...")
    
def load_df(config):
    try:
            full_df = pd.read_csv(config.DATA_CONFIG.get("DATASET_CSV"), index_col=0, parse_dates=True)
            #print(f"Loaded CSV with shape {full_df.shape}.")
            return full_df
    except Exception as e:
            print(f"Error loading CSV File: {e}")
            return
    
def main():
    config = Config()
    signal.signal(signal.SIGINT, signal_handler)
    training_module=None
    try:
        full_df = load_df(config=config)
        print(f'df shape: {full_df.shape}')
        if full_df is None:
            print("Failed to load dataframe. Exiting")
            return
        
        # Initialize environment
        print(full_df.shape)
        env = MarketEnv(data=full_df,
        initial_capital=config.MARKET_ENV_PARMS.get("INITIAL_CAPITAL"),
        max_trades_per_month=config.MARKET_ENV_PARMS.get("MAX_TRADES_PER_MONTH"),
        trading_fee=config.MARKET_ENV_PARMS.get("TRADING_FEE"),
        hold_penalty=config.MARKET_ENV_PARMS.get("HOLD_PENALTY"),
        max_hold_steps=config.MARKET_ENV_PARMS.get("MAX_HOLD_STEPS"),
        segment_size=config.DATA_CONFIG.get("SEGMENT_SIZE"),
        num_projected_days=config.MARKET_ENV_PARMS.get("NUM_PROJECTED_DAYS"))
        

        env.reset()
        # Get initial state to determine dimensions
        initial_state = env.get_state()
        state_dim = initial_state.shape  # shape should be (segment_size, feature_dim)
        action_dim = env.action_space.n
    
        main_model, target_model = initialize_models(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        )
        check_state_dimensions(env, main_model, 'mps')
        training_module = Training(env=env,
        main_model=main_model,
        target_model=target_model,
        config=config
        )
    
        print("\nStarting training...")
    
        trained_model = training_module.train(should_exit_flag=lambda: should_exit)

        if not should_exit:
            
        # Save the trained model
            print("\nSaving model...")
            torch.save({
            'model_state_dict': trained_model.state_dict(),
            'hyperparameters': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'embed_dim': config.TRAINING_PARMS.get('EMBED_DIMENSION'),
            'num_heads': config.TRAINING_PARMS.get('NUM_HEADS'),
            'dropout_rate': config.TRAINING_PARMS.get('DROPOUT_RATE')
            },
        }, 'trained_attention_dqn.pth')

        print("\nTraining complete! Model saved!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        cleanup(training_module, main_model)

if __name__ == "__main__":
    main()