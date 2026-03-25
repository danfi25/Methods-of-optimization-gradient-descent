from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# one optimization run's full history
@dataclass
class OptimizationResult:
    x_history: list[NDArray[np.float64]] = field(default_factory=list)
    f_history: list[float] = field(default_factory=list)
    grad_history: list[NDArray[np.float64]] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)

    x_opt: NDArray[np.float64] | None = None
    f_opt: float | None = None
    n_iter: int = 0
    converged: bool = False
    stop_reason: str = ""

    n_func_evals: int = 0
    n_grad_evals: int = 0

    def record(self, x: NDArray, f: float, grad: NDArray, lr: float):
        self.x_history.append(x.copy())
        self.f_history.append(f)
        self.grad_history.append(grad.copy())
        self.lr_history.append(lr)

    def finalize(self, converged: bool, stop_reason: str):
        # setting final state from the last recorded iteration
        self.x_opt = self.x_history[-1]
        self.f_opt = self.f_history[-1]
        self.n_iter = len(self.x_history) - 1
        self.converged = converged
        self.stop_reason = stop_reason

    @property
    def trajectory(self) -> NDArray[np.float64]:
        return np.array(self.x_history)  # return x_history as (N, d) array for plotting

    @property
    def grad_norms(self) -> NDArray[np.float64]:
        return np.array([np.linalg.norm(g) for g in self.grad_history])  # gradient norms
