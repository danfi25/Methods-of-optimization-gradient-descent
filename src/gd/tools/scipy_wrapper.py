from numpy.typing import NDArray
from typing import Callable
from scipy.optimize import minimize

from src.gd.db.history import OptimizationResult


def scipy_minimize(
        func: Callable[[NDArray], float],
        grad: Callable[[NDArray], NDArray],
        x0: NDArray,
        method: str = "CG",
        **kwargs,
) -> OptimizationResult:
    """
    Wrapper around scipy.optimize.minimize that records full trajectory.

    Supported methods: "CG", "BFGS", "L-BFGS-B".
    """
    result = OptimizationResult()
    counters = {"func": 0, "grad": 0}

    def tracked_func(x):
        counters["func"] += 1
        return func(x)

    def tracked_grad(x):
        counters["grad"] += 1
        return grad(x)

    f0 = func(x0)
    g0 = grad(x0)
    result.record(x0.copy(), f0, g0, lr=0.0)

    def callback(xk):
        fk = func(xk)
        gk = grad(xk)
        result.record(xk.copy(), fk, gk, lr=0.0)

    scipy_result = minimize(
        tracked_func, x0, jac=tracked_grad, method=method,
        callback=callback, **kwargs,
    )

    result.n_func_evals = counters["func"]
    result.n_grad_evals = counters["grad"]
    result.finalize(
        converged=scipy_result.success,
        stop_reason=scipy_result.message,
    )
    return result
