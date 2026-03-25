from typing import Protocol


class LRSchedule(Protocol):
    # protocol for learning rate schedules

    def __call__(self, iteration: int, **kwargs) -> float: ...


class ConstantLR:
    # fixed learning rate

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def __call__(self, iteration: int, **kwargs) -> float:
        return self.lr

    def __repr__(self) -> str:
        return f"ConstantLR(lr={self.lr})"


class ExponentialDecayLR:
    # lr(t) = lr_0 * decay_rate^t

    def __init__(self, lr_0: float = 0.1, decay_rate: float = 0.99):
        self.lr_0 = lr_0
        self.decay_rate = decay_rate

    def __call__(self, iteration: int, **kwargs) -> float:
        return self.lr_0 * self.decay_rate ** iteration

    def __repr__(self) -> str:
        return f"ExponentialDecayLR(lr_0={self.lr_0}, decay_rate={self.decay_rate})"


class InverseDecayLR:
    # lr(t) = lr_0 / (1 + decay * t)

    def __init__(self, lr_0: float = 0.1, decay: float = 0.01):
        self.lr_0 = lr_0
        self.decay = decay

    def __call__(self, iteration: int, **kwargs) -> float:
        return self.lr_0 / (1 + self.decay * iteration)

    def __repr__(self) -> str:
        return f"InverseDecayLR(lr_0={self.lr_0}, decay={self.decay})"


class AdaptiveGradLR:
    # lr(t) = lr_0 / (1 + ||grad||)
    # scales with gradient magnitude

    def __init__(self, lr_0: float = 0.1):
        self.lr_0 = lr_0

    def __call__(self, iteration: int, **kwargs) -> float:
        grad_norm = kwargs.get("grad_norm", 1.0)
        return self.lr_0 / (1 + grad_norm)

    def __repr__(self) -> str:
        return f"AdaptiveGradLR(lr_0={self.lr_0})"
