from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from src.gd.db.history import OptimizationResult
from src.gd.bonus.line_search import armijo_backtracking, wolfe_line_search
from src.gd.settings.lr_schedules import LRSchedule, ConstantLR
from src.gd.settings.stopping import StoppingCriteria, CombinedStopping, GradientNorm


class GradientDescent:

    """
    configurable gradient descent

    call minimize(func, grad, x0) to run the optimization


    lr_schedule : LRSchedule or None
        Learning rate schedule. Ignored when step_size_method != "fixed"
    step_size_method : "fixed", "armijo", or "wolfe"
        How to choose the step size each iteration based on the type of algorithm
    stopping : StoppingCriteria or None
        Stopping criteria. Defaults to gradient norm < 1e-6
    max_iter : int
        Maximum number of iterations
    **ls_kwargs
        Extra arguments forwarded to the line search
        (c1, c2, rho, alpha_init)
    """
    def __init__(
        self,
        lr_schedule: LRSchedule | None = None,
        step_size_method: Literal["fixed", "armijo", "wolfe"] = "fixed",
        stopping: StoppingCriteria | None = None,
        max_iter: int = 10000,
        **ls_kwargs,
    ):
        self.lr_schedule = lr_schedule or ConstantLR(0.01)
        self.step_size_method = step_size_method
        self.max_iter = max_iter
        self.ls_kwargs = ls_kwargs

        if stopping is not None:
            self.stopping = stopping
        else:
            self.stopping = CombinedStopping([GradientNorm(1e-6)], max_iter=max_iter)

    def minimize(
        self,
        func: Callable[[NDArray], float],
        grad: Callable[[NDArray], NDArray],
        x0: NDArray,
    ) -> OptimizationResult:
        # run gradient descent and returns full optimization history
        if self.step_size_method == "fixed":
            return self._run_fixed(func, grad, x0)
        return self._run_line_search(func, grad, x0)


    def _run_fixed(
        self,
        func: Callable[[NDArray], float],
        grad: Callable[[NDArray], NDArray],
        x0: NDArray,
    ) -> OptimizationResult:
        result = OptimizationResult()
        x = np.array(x0, dtype=np.float64)

        f_val = func(x)
        g = grad(x)
        result.n_func_evals += 1
        result.n_grad_evals += 1

        lr = self.lr_schedule(0, grad_norm=np.linalg.norm(g))
        result.record(x, f_val, g, lr)

        for i in range(1, self.max_iter + 1):
            lr = self.lr_schedule(i, grad_norm=np.linalg.norm(g))
            x = x - lr * g

            f_prev = f_val
            f_val = func(x)
            g = grad(x)
            result.n_func_evals += 1
            result.n_grad_evals += 1

            result.record(x, f_val, g, lr)

            if not np.isfinite(f_val):
                result.finalize(converged=False, stop_reason="diverged (f became inf/nan)")
                return result

            stop, reason = self.stopping.check(i, x, f_val, g, f_prev)
            if stop:
                result.finalize(
                    converged="max iterations" not in reason,
                    stop_reason=reason,
                )
                return result

        result.finalize(converged=False, stop_reason="max iterations reached (fallback)")
        return result

    def _run_line_search(
        self,
        func: Callable[[NDArray], float],
        grad: Callable[[NDArray], NDArray],
        x0: NDArray,
    ) -> OptimizationResult:
        result = OptimizationResult()
        x = np.array(x0, dtype=np.float64)

        f_val = func(x)
        g = grad(x)
        result.n_func_evals += 1
        result.n_grad_evals += 1
        result.record(x, f_val, g, lr=0.0)

        for i in range(1, self.max_iter + 1):
            direction = -g

            if self.step_size_method == "armijo":
                alpha, fe = armijo_backtracking(
                    func, x, g, direction, **self.ls_kwargs,
                )
                result.n_func_evals += fe
            elif self.step_size_method == "wolfe":
                alpha, fe, ge = wolfe_line_search(
                    func, grad, x, g, direction, **self.ls_kwargs,
                )
                result.n_func_evals += fe
                result.n_grad_evals += ge
            else:
                raise ValueError(f"Unknown step_size_method: {self.step_size_method}")

            x = x + alpha * direction
            f_prev = f_val
            f_val = func(x)
            g = grad(x)
            result.n_func_evals += 1
            result.n_grad_evals += 1
            result.record(x, f_val, g, lr=alpha)

            stop, reason = self.stopping.check(i, x, f_val, g, f_prev)
            if stop:
                result.finalize(
                    converged="max iterations" not in reason,
                    stop_reason=reason,
                )
                return result

        result.finalize(converged=False, stop_reason="max iterations reached")
        return result

    def __repr__(self) -> str:
        if self.step_size_method == "fixed":
            return f"GradientDescent(lr={self.lr_schedule}, max_iter={self.max_iter})"
        return (
            f"GradientDescent(step_size_method='{self.step_size_method}', "
            f"max_iter={self.max_iter})"
        )
