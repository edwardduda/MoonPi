# classes/ReplayBuffer.py

import numpy as np

class ReplayBuffer:
    def __init__(self, state_shape, max_size):
        self.max_size    = max_size
        self.ptr         = 0
        self.full        = False
        self.state_shape = state_shape  # e.g. (70, 179)

        # now these all match: (max_size, 70, 179)
        self.states      = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions     = np.zeros(max_size, dtype=np.int64)
        self.rewards     = np.zeros(max_size, dtype=np.float32)
        self.dones       = np.zeros(max_size, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = done

        self.ptr = (self.ptr + 1) % self.max_size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_idx = self.max_size if self.full else self.ptr
        idxs    = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.states[idxs],      # (B, 70, 179)
            self.actions[idxs],     # (B,)
            self.rewards[idxs],     # (B,)
            self.next_states[idxs], # (B, 70, 179)
            self.dones[idxs]        # (B,)
        )

    def __len__(self):
        return self.max_size if self.full else self.ptr
