import numpy as np

from tests.benchmarks import BENCHMARKS_2D
from src.gd.bonus.line_search import armijo_backtracking, wolfe_line_search


def test_armijo_condition_holds():
    # returned step size should satisfy the Armijo condition
    bench = BENCHMARKS_2D["rosenbrock"]
    x = np.array([-1.0, 1.0])
    g = bench.grad(x)
    d = -g
    alpha, _ = armijo_backtracking(bench.func, x, g, d, c1=1e-4)

    f0 = bench.func(x)
    f_new = bench.func(x + alpha * d)
    assert f_new <= f0 + 1e-4 * alpha * np.dot(g, d)


def test_wolfe_conditions_hold():
    # returned step size should satisfy both Wolfe conditions
    bench = BENCHMARKS_2D["sphere"]
    x = np.array([3.0, 4.0])
    g = bench.grad(x)
    d = -g
    c1, c2 = 1e-4, 0.9

    alpha, _, _ = wolfe_line_search(bench.func, bench.grad, x, g, d, c1=c1, c2=c2)

    f0 = bench.func(x)
    f_new = bench.func(x + alpha * d)
    g_new = bench.grad(x + alpha * d)

    # armijo condition
    assert f_new <= f0 + c1 * alpha * np.dot(g, d)
    # curvature condition (strong Wolfe)
    assert abs(np.dot(g_new, d)) <= c2 * abs(np.dot(g, d))
