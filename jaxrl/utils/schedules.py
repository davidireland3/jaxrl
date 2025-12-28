class LinearSchedule:
    """Linear schedule from initial_value to final_value over duration steps."""

    def __init__(self, initial_value: float, final_value: float, duration: int) -> None:
        self.initial_value = initial_value
        self.final_value = final_value
        self.duration = duration

    def __call__(self, step: int) -> float:
        fraction = min(step / self.duration, 1.0)
        return self.initial_value + fraction * (self.final_value - self.initial_value)


class ExponentialSchedule:
    """Exponential decay: value = initial_value * (decay_rate ** (step / decay_steps))"""

    def __init__(self, initial_value: float, decay_rate: float, decay_steps: int = 1) -> None:
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        return self.initial_value * (self.decay_rate ** (step / self.decay_steps))
