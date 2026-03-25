from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class StoppingCriteria(Protocol):
    # returns(should_stop, reason)

    def check(
        self, iteration: int, x: NDArray, f: float, grad: NDArray, f_prev: float | None
    ) -> tuple[bool, str]: ...


class MaxIterations:
    def __init__(self, max_iter: int = 10000):
        self.max_iter = max_iter

    def check(self, iteration, x, f, grad, f_prev) -> tuple[bool, str]:
        if iteration >= self.max_iter:
            return True, f"max iterations ({self.max_iter}) reached"
        return False, ""


class GradientNorm:
    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def check(self, iteration, x, f, grad, f_prev) -> tuple[bool, str]:
        norm = np.linalg.norm(grad)
        if norm < self.tol:
            return True, f"gradient norm {norm:.2e} < {self.tol:.2e}"
        return False, ""


class FunctionValueChange:
    def __init__(self, tol: float = 1e-10):
        self.tol = tol

    def check(self, iteration, x, f, grad, f_prev) -> tuple[bool, str]:
        if f_prev is not None and abs(f - f_prev) < self.tol:
            return True, f"|f_k - f_{{k-1}}| = {abs(f - f_prev):.2e} < {self.tol:.2e}"
        return False, ""


class CombinedStopping:
    # Stop on any criteria

    def __init__(self, criteria: list, max_iter: int = 10000):
        self.criteria = criteria + [MaxIterations(max_iter)]

    def check(self, iteration, x, f, grad, f_prev) -> tuple[bool, str]:
        for criterion in self.criteria:
            stop, reason = criterion.check(iteration, x, f, grad, f_prev)
            if stop:
                return True, reason
        return False, ""
