import math

class EpsilonSchedule:
    """
    • warmup_steps : keep ε == start for this many steps  
    • start        : initial ε plateau (e.g. 1.0)  
    • end          : minimum ε floor (e.g. 0.25)  
    • reset        : max ε after floor (e.g. 0.35)  
    • period       : length of one cosine cycle after warm-up  
    """
    def __init__(self, warmup_steps, start, end, reset, period):
        self.warmup   = warmup_steps
        self.start    = start
        self.end      = end
        self.reset    = reset
        self.period   = period
        self._counter = 0

        # last computed value (so we can “peek” without advancing)
        self.current  = start

    def step(self) -> float:
        """Advance the schedule by one step and return ε_t."""
        t = self._counter
        if t < self.warmup:
            eps = self.start
        else:
            # shift so t' = 0 right after warm-up
            tp    = (t - self.warmup) % self.period
            phase = (math.pi * tp) / self.period       # 0 → π
            eps   = self.end + (self.reset - self.end) * 0.5 * (1 + math.cos(phase))
        self._counter += 1
        self.current   = eps
        return eps
