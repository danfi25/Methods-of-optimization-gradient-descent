from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.gd.db.history import OptimizationResult


class Visualizer:
    """
    Generic optimization visualizer.

    Works with any optimizer that produces OptimizationResult objects.
    Accepts the objective function and domain directly, so it is not
    tied to any specific benchmark registry.

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float (used for surface/contour grids).
    domain : tuple[float, float]
        Symmetric plot bounds, e.g. (-5, 5).
    known_minima : list of (x_min, f_min) tuples, optional
        Marked with red crosses on contour plots.
    title : str, optional
        Base title used in plots.

    Examples
    --------
    >>> vis = Visualizer(func=rosenbrock, domain=(-3, 3), title="Rosenbrock")
    >>> vis.contour({"GD": result1, "BFGS": result2})
    >>> vis.surface_3d({"GD": result1})
    >>> vis.convergence({"GD": result1}, metric="f_value")
    >>> print(vis.table({"GD": result1}))
    """

    def __init__(
        self,
        func: Callable,
        domain: tuple[float, float],
        known_minima: list[tuple[np.ndarray, float]] | None = None,
        title: str = "",
    ):
        self.func = func
        self.domain = domain
        self.known_minima = known_minima or []
        self.title = title

    def _grid(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute meshgrid and Z values."""
        lo, hi = self.domain
        x = np.linspace(lo, hi, n)
        y = np.linspace(lo, hi, n)
        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [[self.func(np.array([xi, yi])) for xi in x] for yi in y]
        )
        return X, Y, Z

    def contour(
        self,
        results: dict[str, OptimizationResult],
        n_grid: int = 200,
        figsize: tuple[float, float] = (10, 8),
        levels: int = 50,
        log_scale: bool = True,
    ) -> Figure:
        """2D contour plot with optimization trajectories overlaid."""
        X, Y, Z = self._grid(n_grid)

        fig, ax = plt.subplots(figsize=figsize)

        if log_scale:
            Z_plot = np.log1p(Z - Z.min())
            ax.contourf(X, Y, Z_plot, levels=levels, cmap="viridis", alpha=0.7)
            ax.contour(
                X, Y, Z_plot, levels=levels // 2,
                colors="gray", linewidths=0.3, alpha=0.5,
            )
        else:
            ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.7)

        colors = plt.cm.tab10.colors
        for idx, (label, res) in enumerate(results.items()):
            traj = res.trajectory
            color = colors[idx % len(colors)]
            ax.plot(
                traj[:, 0], traj[:, 1], "o-",
                color=color, markersize=2, linewidth=1.2, label=label,
            )
            ax.plot(traj[0, 0], traj[0, 1], "s", color=color, markersize=8)
            ax.plot(traj[-1, 0], traj[-1, 1], "*", color=color, markersize=12)

        for x_min, f_min in self.known_minima:
            ax.plot(x_min[0], x_min[1], "rx", markersize=14, markeredgewidth=3)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{self.title} — Optimization Trajectories")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def surface_3d(
        self,
        results: dict[str, OptimizationResult],
        n_grid: int = 100,
        figsize: tuple[float, float] = (12, 8),
        elev: float = 35,
        azim: float = -60,
    ) -> Figure:
        """3D surface plot with trajectories drawn on the surface."""
        X, Y, Z = self._grid(n_grid)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5, edgecolor="none")

        colors = plt.cm.tab10.colors
        for idx, (label, res) in enumerate(results.items()):
            traj = res.trajectory
            f_traj = np.array(res.f_history)
            color = colors[idx % len(colors)]
            ax.plot(
                traj[:, 0], traj[:, 1], f_traj, "o-",
                color=color, markersize=2, linewidth=1.5,
                label=label, zorder=10,
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.set_title(f"{self.title} — 3D Trajectories")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    @staticmethod
    def convergence(
        results: dict[str, OptimizationResult],
        metric: str = "f_value",
        figsize: tuple[float, float] = (10, 6),
        log_y: bool = True,
        title: str | None = None,
    ) -> Figure:
        """
        Convergence plot: metric vs iteration number.

        This is a static method — it only needs OptimizationResult objects,
        no function or domain info.

        metric : "f_value", "grad_norm", or "lr"
        """
        fig, ax = plt.subplots(figsize=figsize)
        ylabel_map = {
            "f_value": "f(x)",
            "grad_norm": "||grad f(x)||",
            "lr": "Learning rate",
        }

        for label, res in results.items():
            if metric == "f_value":
                y_data = res.f_history
            elif metric == "grad_norm":
                y_data = res.grad_norms.tolist()
            elif metric == "lr":
                y_data = res.lr_history
            else:
                raise ValueError(f"Unknown metric: {metric}")
            ax.plot(range(len(y_data)), y_data, label=label)

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.set_title(title or f"Convergence — {ylabel_map.get(metric, metric)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def table(
        results: dict[str, OptimizationResult],
        known_minimum: float = 0.0,
    ) -> str:
        """Generate a Markdown comparison table."""
        header = (
            "| Method | x_opt | f_opt | |f - f*| | Iters "
            "| Func evals | Grad evals | Converged | Stop reason |"
        )
        separator = "|---|---|---|---|---|---|---|---|---|"
        rows = [header, separator]

        for label, res in results.items():
            x_str = np.array2string(res.x_opt, precision=4, suppress_small=True)
            f_str = f"{res.f_opt:.6e}"
            err_str = f"{abs(res.f_opt - known_minimum):.2e}"
            row = (
                f"| {label} | {x_str} | {f_str} | {err_str} "
                f"| {res.n_iter} | {res.n_func_evals} | {res.n_grad_evals} "
                f"| {'Yes' if res.converged else 'No'} | {res.stop_reason} |"
            )
            rows.append(row)

        return "\n".join(rows)
