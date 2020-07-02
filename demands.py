import numpy as np


class DemandGenerator:

    def __init__(self, demands_pattern='classic_beer_game', size=None, low=None, high=None, 
                 mean=None, sd=None):
        """
        Args:
            demands_pattern: can be a string ('normal', 'uniform', 'classic_beer_game') to specify a stochastic demand
                pattern or a list of numbers to specify deterministic demands.
            low: must be provided when uniform pattern is specified
            high: must be provided when uniform pattern is specified
            mean: mean of a normal distribution, must be provided when normal pattern is specified
            sd: standard deviation of a normal distribution, must be provided when normal pattern is specified
        """
        self.demands_pattern = demands_pattern
        self.size = size
        self.low = low
        self.high = high
        self.mean = mean
        self.sd = sd
        self.reset()

    def reset(self):
        if isinstance(self.demands_pattern, (np.ndarray, list)):
            self.demands = self.demands_pattern

        elif self.demands_pattern == 'uniform':
            if (self.low is None) or (self.high is None):
                raise ValueError('"low" and "high" need to be provided when uniform pattern is specified')
            self.demands = np.random.randint(self.low, self.high, self.size)

        elif self.demands_pattern == 'normal':
            self.demands = (self.mean + self.sd * np.random.randn(self.size)).astype(int)
            if (self.mean is None) or (self.sd is None):
                raise ValueError('"mean" and "sd" need to be provided when normal pattern is specified')

        elif self.demands_pattern == 'classic_beer_game':
            self.demands = np.array([4] * 4 + [8] * 31)

        else:
            raise ValueError("Demand pattern not recognized")

    def get_demand(self, period):
        return self.demands[period].item()
