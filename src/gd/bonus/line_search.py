import numpy as np
from numpy.typing import NDArray
from typing import Callable

from src.gd.db.history import OptimizationResult
from src.gd.settings.stopping import StoppingCriteria, CombinedStopping, GradientNorm


# armijo backtracking line search
def armijo_backtracking(
        func: Callable,
        x: NDArray,
        grad: NDArray,
        direction: NDArray,
        alpha_init: float = 1.0,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_backtracks: int = 50,
) -> tuple[float, int]:
    alpha = alpha_init  # start from 1
    f0 = func(x)
    slope = np.dot(grad, direction)
    evals = 1

    for _ in range(max_backtracks):
        if func(x + alpha * direction) <= f0 + c1 * alpha * slope:
            evals += 1
            return alpha, evals
        evals += 1
        alpha *= rho  # cut in half every time we're not done

    return alpha, evals  # returns (step_size, n_func_evals)


def wolfe_line_search(
        func: Callable,
        grad_func: Callable,
        x: NDArray,
        grad: NDArray,
        direction: NDArray,
        alpha_init: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 50,
) -> tuple[float, int, int]:
    f0 = func(x)
    slope0 = np.dot(grad, direction)
    func_evals, grad_evals = 1, 0

    alpha_lo, alpha_hi = 0.0, alpha_init * 4
    alpha = alpha_init

    for _ in range(max_iter):
        f_new = func(x + alpha * direction)
        func_evals += 1

        if f_new > f0 + c1 * alpha * slope0:
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2
            continue

        g_new = grad_func(x + alpha * direction)
        grad_evals += 1
        slope_new = np.dot(g_new, direction)

        if abs(slope_new) <= c2 * abs(slope0):  # same deal as the previous one but keeping the slope steepness as well
            return alpha, func_evals, grad_evals

        if slope_new > 0:
            alpha_hi = alpha
        else:
            alpha_lo = alpha
        alpha = (alpha_lo + alpha_hi) / 2

    return alpha, func_evals, grad_evals  # returns (step_size, n_func_evals, n_grad_evals)
