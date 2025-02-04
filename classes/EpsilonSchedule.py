class EpsilonSchedule:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.epsilon = start
        self.steps = 0
        
    def step(self) -> float:
        if self.epsilon == 0.6:
            self.decay == 0.999995
        self.epsilon = max(self.end, self.epsilon * self.decay)
        self.steps += 1
        
        if(self.epsilon == self.end):
            self.epsilon = 0.3
        return self.epsilon
    
    def reset(self):
        """Reset epsilon to initial value"""
        self.epsilon = self.start
        self.steps = 0
    
    @property
    def exploration_rate(self) -> float:
        """Return current exploration rate as percentage"""
        return self.epsilon * 100