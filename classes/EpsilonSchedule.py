import math
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def step(t, warmup, period, end, eps):
    tp    = (t - warmup) % period
    phase = (2 * math.pi * tp) / period
    eps   = end + (eps - end) * 0.5 * (1 + math.cos(phase))
    return eps

class EpsilonSchedule:
    """
    • warmup_steps : keep ε == start for this many steps  
    • start        : initial ε plateau (e.g. 1.0)  
    • end          : minimum ε floor (e.g. 0.25)  
    • reset        : max ε after floor (e.g. 0.35)  
    • period       : length of one cosine cycle after warm-up  
    """
    def __init__(self, warmup_steps, start, end, reset, period):
        self.warmup_steps   = warmup_steps
        self.start    = start
        self.end      = end
        self.reset    = reset
        self.period   = period
        self.counter = 0
        self.eps = 1.0
        self.phase = None
        # last computed value (so we can “peek” without advancing)
        
    
    def step(self):

        while self.counter < self.warmup_steps:
            self.eps = self.start
            self.counter += 1
        if self.eps > self.reset:
            
            self.eps = step(self.counter, self.warmup_steps, self.period, self.end, self.eps)
        else:
            self.eps = step(self.counter, self.warmup_steps, self.period, self.end, self.reset)

        self.counter += 1

        return self.eps
