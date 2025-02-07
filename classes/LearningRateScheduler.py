import torch
from torch.optim.lr_scheduler import _LRScheduler

class EpsilonMatchingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, 
                 initial_lr, 
                 min_lr,      
                 warmup_steps,
                 epsilon_decay,
                 epsilon_min,
                 last_epoch=-1):
        # Store base parameters
        self.base_initial_lr = initial_lr  # Reference if needed
        self.initial_lr = initial_lr       # Current cycle's maximum lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # This offset marks the beginning of the current cycle (post-warmup)
        self.cycle_offset = 0  
        self.current_lr = initial_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # Warmup phase for the very first cycle
        if self._step_count < self.warmup_steps:
            warmup_factor = float(self._step_count) / float(max(1, self.warmup_steps))
            lr = self.initial_lr * warmup_factor
        else:
            # Calculate steps since current cycle started (post-warmup)
            cycle_steps = self._step_count - self.warmup_steps - self.cycle_offset
            # Compute decay factor and ensure it does not drop below epsilon_min
            decay_factor = max(self.epsilon_min, self.epsilon_decay ** float(cycle_steps))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * decay_factor
            
            # If we've effectively reached the minimum lr, reset the cycle
            if lr <= self.min_lr + 1e-8:  # Allow for floating point tolerance
                self.initial_lr = self.initial_lr * 0.8  # Decay the maximum lr for next cycle
                self.cycle_offset = self._step_count - self.warmup_steps
                lr = self.initial_lr  # Restart cycle with new max lr

        self.current_lr = float(lr)
        return [self.current_lr for _ in self.optimizer.param_groups]

    def get_last_lr(self):
        return [self.current_lr for _ in self.optimizer.param_groups]