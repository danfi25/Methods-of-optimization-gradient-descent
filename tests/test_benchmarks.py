import numpy as np
import pytest

from tests.benchmarks import BENCHMARKS_2D, numerical_gradient


@pytest.mark.parametrize("name,bench", list(BENCHMARKS_2D.items()))
def test_known_minima(name, bench):
    # verify that f(x_min) equals the known minimum value
    for x_min, f_min in bench.known_minima:
        assert bench.func(x_min) == pytest.approx(f_min, abs=1e-8), (
            f"{name}: f({x_min}) = {bench.func(x_min)}, expected {f_min}"
        )


@pytest.mark.parametrize("name,bench", list(BENCHMARKS_2D.items()))
def test_gradient_correctness(name, bench):
    # verify that analytical gradient matches numerical gradient at random points
    rng = np.random.default_rng(42)
    lo, hi = bench.domain
    for _ in range(5):
        x = rng.uniform(lo * 0.5, hi * 0.5, size=bench.n_dims)
        analytical = bench.grad(x)
        numerical = numerical_gradient(bench.func, x)
        np.testing.assert_allclose(
            analytical, numerical, atol=1e-5,
            err_msg=f"{name}: gradient mismatch at x={x}",
        )
