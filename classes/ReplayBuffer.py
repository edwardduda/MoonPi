import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _update_sum_tree(tree, idx, priority):
    """Update a sum tree with a new priority value."""
    change = priority - tree[idx]
    tree[idx] = priority
    
    # Propagate changes upward through parent nodes
    while idx > 0:
        idx = (idx - 1) // 2
        tree[idx] += change
    
    return tree

@jit(nopython=True, cache=True)
def _sample_sum_tree(tree, batch_size, max_idx):
    """Sample indices from the sum tree based on priorities."""
    indices = np.zeros(batch_size, dtype=np.int64)
    tree_length = len(tree)
    # Get total priority from root node
    total_priority = tree[0]
    if total_priority <= 0:
        # Fallback to uniform sampling if tree has no priority
        # Use standard randint without dtype parameter
        for i in range(batch_size):
            indices[i] = np.random.randint(0, max_idx)
        return indices
    
    # Divide total priority into segments
    segment = total_priority / batch_size
    
    for i in range(batch_size):
        # Sample uniformly within each segment
        a = segment * i
        b = segment * (i + 1)
        value = np.random.uniform(a, b)
        
        # Traverse tree to find corresponding index
        idx = 0  # Start at root
        while 2 * idx + 1 < tree_length:  # While not at a leaf node
            left = 2 * idx + 1
            right = left + 1
            
            if right >= tree_length or value <= tree[left]:
                idx = left
            else:
                value -= tree[left]
                idx = right
                
        # Convert tree index to data index
        data_idx = idx - (tree_length // 2)
        
        # Safety bounds check
        if data_idx < 0:
            data_idx = 0
        elif data_idx >= max_idx:
            data_idx = max_idx - 1
            
        indices[i] = data_idx
    
    return indices

class ReplayBuffer:
    def __init__(self, state_shape, max_size, alpha=0.6, beta_start=0.4, beta_increment=0.001, epsilon=1e-5):
        self.max_size = max_size
        self.ptr = 0
        self.full = False
        self.state_shape = state_shape
        
        # PER hyperparameters
        self.alpha = alpha  # Priority exponent - how much to prioritize (0 = uniform, 1 = full prioritization)
        self.beta = beta_start  # Importance sampling weight to correct bias
        self.beta_increment = beta_increment  # Gradually increase beta to 1
        self.epsilon = epsilon  # Small constant to ensure non-zero priority
        self.max_priority = 1.0  # Initial priority for new experiences
        
        # Sum tree for efficient sampling (binary tree with size 2*max_size-1)
        self.tree_size = 2 * max_size - 1
        self.sum_tree = np.zeros(self.tree_size, dtype=np.float32)
        
        # Experience storage
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.bool_)
        
    def push(self, state, action, reward, next_state, done):
        """Add new experience with maximum priority to ensure it gets sampled."""
        idx = self.ptr
        
        # Store experience
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        # Calculate tree index for this experience
        tree_idx = idx + self.tree_size // 2
        
        # Assign max priority for new experiences to ensure they get sampled
        priority = self.max_priority ** self.alpha
        self.sum_tree = _update_sum_tree(self.sum_tree, tree_idx, priority)
        
        # Update pointer and full flag
        self.ptr = (self.ptr + 1) % self.max_size
        if self.ptr == 0:
            self.full = True
            
        # Gradually increase beta to reduce importance sampling correction over time
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def sample(self, batch_size):
        """Sample batch of experiences based on their priorities."""
        max_idx = self.max_size if self.full else self.ptr
        
        if max_idx == 0:
            raise ValueError("Cannot sample from an empty buffer")
        
        # Sample indices based on priority
        indices = _sample_sum_tree(self.sum_tree, batch_size, max_idx)
        
        # Update priorities based on reward magnitude (proxy for TD error)
        # In ideal implementation, we'd use actual TD errors from the network
        for idx in indices:
            tree_idx = idx + self.tree_size // 2
            
            # Use reward magnitude as approximation of experience importance
            # Experiences with larger rewards (positive or negative) are likely more important
            priority = (abs(self.rewards[idx]) + self.epsilon) ** self.alpha
            self.sum_tree = _update_sum_tree(self.sum_tree, tree_idx, priority)
        
        # Return experiences with same interface as original ReplayBuffer
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        """Return the current number of experiences in the buffer."""
        return self.max_size if self.full else self.ptr