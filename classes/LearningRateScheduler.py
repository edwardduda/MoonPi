import torch
from torch.optim.lr_scheduler import _LRScheduler


class EpsilonMatchingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, 
                 initial_lr,  # Remove default value
                 min_lr,      # Remove default value
                 warmup_steps,
                 epsilon_decay,
                 epsilon_min,
                 last_epoch=-1):
        
        # Store all parameters before super().__init__
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_lr = initial_lr

        # Call parent class init with base parameters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # Make sure we have valid values
        if self._step_count is None:
            return [self.initial_lr for _ in self.optimizer.param_groups]

        # Warmup phase
        if self._step_count < self.warmup_steps:
            warmup_factor = float(self._step_count) / float(max(1, self.warmup_steps))
            lr = self.initial_lr * warmup_factor
        else:
            # Calculate steps since warmup
            post_warmup_steps = self._step_count - self.warmup_steps
            
            # Calculate decay factor similar to epsilon
            decay_factor = max(
                self.epsilon_min,
                (self.epsilon_decay ** float(post_warmup_steps))
            )
            
            # Scale learning rate between initial and minimum
            lr_range = self.initial_lr - self.min_lr
            lr = self.min_lr + (lr_range * decay_factor)

        self.current_lr = float(lr)  # Ensure we're working with floats
        return [self.current_lr for _ in self.optimizer.param_groups]

    def get_last_lr(self):
        return [self.current_lr for _ in self.optimizer.param_groups]