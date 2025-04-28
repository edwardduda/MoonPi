import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, min_lr: float,
                 period: int, last_epoch: int = -1):
        self.base_max_lr = initial_lr
        self.min_lr      = min_lr
        self.period      = period
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t       = self.last_epoch % self.period
        cosine  = 0.5 * (1 + math.cos(2.0 * math.pi * t / self.period))
        return [
            self.min_lr + (self.base_max_lr - self.min_lr) * cosine
            for _ in self.optimizer.param_groups
        ]
