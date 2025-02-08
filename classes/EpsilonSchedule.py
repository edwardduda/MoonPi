class EpsilonSchedule:
    def __init__(self, start, end, decay, reset):
        self.start = start
        self.end = end
        self.decay = decay
        self.epsilon = start
        self.steps = 0
        self.epsilon_reset = reset
        
    def step(self) -> float:
        # Decay epsilon monotonically until it reaches the floor (end)
        self.epsilon = max(self.end, self.epsilon * self.decay)
        self.steps += 1
        
        if self.epsilon == self.end:
            self.epsilon = self.epsilon_reset
        return self.epsilon
    
    def reset(self):
        """Reset epsilon to its initial value."""
        self.epsilon = self.start
        self.steps = 0
    
    @property
    def exploration_rate(self) -> float:
        """Return the current exploration rate as a percentage."""
        return self.epsilon * 100