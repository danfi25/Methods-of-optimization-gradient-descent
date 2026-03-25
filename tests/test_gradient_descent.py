import numpy as np

from tests.benchmarks import BENCHMARKS_2D
from src.gd.gradient_descent import GradientDescent
from src.gd.settings.lr_schedules import ConstantLR, ExponentialDecayLR


def test_sphere_convergence():
    # GD should converge to [0, 0] on the sphere function
    bench = BENCHMARKS_2D["sphere"]
    gd = GradientDescent(lr_schedule=ConstantLR(0.1), max_iter=1000)
    result = gd.minimize(bench.func, bench.grad, x0=np.array([5.0, 5.0]))
    assert result.converged
    np.testing.assert_allclose(result.x_opt, [0.0, 0.0], atol=1e-4)
    assert result.f_opt < 1e-8


def test_booth_convergence():
    # GD should converge to [1, 3] on the Booth function
    bench = BENCHMARKS_2D["booth"]
    gd = GradientDescent(lr_schedule=ExponentialDecayLR(0.05, 0.999), max_iter=10000)
    result = gd.minimize(bench.func, bench.grad, x0=np.array([0.0, 0.0]))
    assert result.converged
    np.testing.assert_allclose(result.x_opt, [1.0, 3.0], atol=1e-3)


def test_result_history_nonempty():
    # OptimizationResult should have non-empty trajectory
    bench = BENCHMARKS_2D["sphere"]
    gd = GradientDescent(lr_schedule=ConstantLR(0.1), max_iter=100)
    result = gd.minimize(bench.func, bench.grad, x0=np.array([1.0, 1.0]))
    assert result.n_iter > 0
    assert len(result.x_history) == result.n_iter + 1
    assert result.trajectory.shape == (result.n_iter + 1, 2)
    assert result.n_func_evals > 0
    assert result.n_grad_evals > 0


def test_armijo_via_class():
    # GD with Armijo line search
    bench = BENCHMARKS_2D["sphere"]
    gd = GradientDescent(step_size_method="armijo", max_iter=1000)
    result = gd.minimize(bench.func, bench.grad, x0=np.array([5.0, 5.0]))
    assert result.converged
    np.testing.assert_allclose(result.x_opt, [0.0, 0.0], atol=1e-4)


def test_wolfe_via_class():
    # GD with Wolfe line search
    bench = BENCHMARKS_2D["booth"]
    gd = GradientDescent(step_size_method="wolfe", max_iter=1000)
    result = gd.minimize(bench.func, bench.grad, x0=np.array([0.0, 0.0]))
    assert result.converged
    np.testing.assert_allclose(result.x_opt, [1.0, 3.0], atol=1e-3)


def test_repr():
    gd1 = GradientDescent(lr_schedule=ConstantLR(0.05))
    assert "GradientDescent" in repr(gd1)
    gd2 = GradientDescent(step_size_method="armijo")
    assert "armijo" in repr(gd2)
