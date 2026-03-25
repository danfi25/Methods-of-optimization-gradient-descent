from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BenchmarkFunction:  # benchmark functions with metadata

    name: str
    func: Callable[[NDArray[np.float64]], float]
    grad: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    n_dims: int
    known_minima: list[tuple[NDArray[np.float64], float]]
    domain: tuple[float, float]



def rosenbrock(x: NDArray):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: NDArray) -> NDArray:
    dx = -2 * (1 - x[0]) + 100 * 2 * (x[1] - x[0] ** 2) * (-2 * x[0])
    dy = 100 * 2 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


def rastrigin(x: NDArray):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rastrigin_grad(x: NDArray) -> NDArray:
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)


def himmelblau(x: NDArray):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_grad(x: NDArray) -> NDArray:
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])


def booth(x: NDArray):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def booth_grad(x: NDArray) -> NDArray:
    dx = 2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5)
    dy = 4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
    return np.array([dx, dy])


def beale(x: NDArray):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


def beale_grad(x: NDArray) -> NDArray:
    t1 = 1.5 - x[0] + x[0] * x[1]
    t2 = 2.25 - x[0] + x[0] * x[1] ** 2
    t3 = 2.625 - x[0] + x[0] * x[1] ** 3
    dx = 2 * t1 * (-1 + x[1]) + 2 * t2 * (-1 + x[1] ** 2) + 2 * t3 * (-1 + x[1] ** 3)
    dy = 2 * t1 * x[0] + 2 * t2 * 2 * x[0] * x[1] + 2 * t3 * 3 * x[0] * x[1] ** 2
    return np.array([dx, dy])


def sphere(x: NDArray) -> float:
    return float(np.sum(x ** 2))


def sphere_grad(x: NDArray) -> NDArray:
    return 2 * x

# usecase
BENCHMARKS_2D: dict[str, BenchmarkFunction] = {
    "rosenbrock": BenchmarkFunction(
        name="Rosenbrock",
        func=rosenbrock,
        grad=rosenbrock_grad,
        n_dims=2,
        known_minima=[(np.array([1.0, 1.0]), 0.0)],
        domain=(-3, 3),
    ),
    "rastrigin": BenchmarkFunction(
        name="Rastrigin",
        func=rastrigin,
        grad=rastrigin_grad,
        n_dims=2,
        known_minima=[(np.array([0.0, 0.0]), 0.0)],
        domain=(-5.12, 5.12),
    ),
    "himmelblau": BenchmarkFunction(
        name="Himmelblau",
        func=himmelblau,
        grad=himmelblau_grad,
        n_dims=2,
        known_minima=[
            (np.array([3.0, 2.0]), 0.0),
            (np.array([-2.805118, 3.131312]), 0.0),
            (np.array([-3.779310, -3.283186]), 0.0),
            (np.array([3.584428, -1.848126]), 0.0),
        ],
        domain=(-5, 5),
    ),
    "booth": BenchmarkFunction(
        name="Booth",
        func=booth,
        grad=booth_grad,
        n_dims=2,
        known_minima=[(np.array([1.0, 3.0]), 0.0)],
        domain=(-10, 10),
    ),
    "beale": BenchmarkFunction(
        name="Beale",
        func=beale,
        grad=beale_grad,
        n_dims=2,
        known_minima=[(np.array([3.0, 0.5]), 0.0)],
        domain=(-4.5, 4.5),
    ),
    "sphere": BenchmarkFunction(
        name="Sphere",
        func=sphere,
        grad=sphere_grad,
        n_dims=2,
        known_minima=[(np.array([0.0, 0.0]), 0.0)],
        domain=(-5, 5),
    ),
}


# bonus
def noisy_rosenbrock(x: NDArray, noise_std: float = 0.1):
    # rosenbrock with additive Gaussian noise
    return rosenbrock(x) + np.random.normal(0, noise_std)


def parametric_rastrigin(x: NDArray, A: float = 10.0) -> float:
    # rastrigin with tunable amplitude A
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def parametric_rastrigin_grad(x: NDArray, A: float = 10.0) -> NDArray:
    return 2 * x + A * 2 * np.pi * np.sin(2 * np.pi * x)


def rosenbrock_nd(x: NDArray) -> float:
    # n-dimensional Rosenbrock
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosenbrock_nd_grad(x: NDArray) -> NDArray:
    n = len(x)
    grad = np.zeros(n)
    for i in range(n - 1):
        grad[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        grad[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return grad


# utility

def numerical_gradient(func: Callable, x: NDArray, eps: float = 1e-7) -> NDArray:
    # central difference gradient approximation
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        grad[i] = (func(x + e) - func(x - e)) / (2 * eps)
    return grad


def hessian_numerical(func: Callable, x: NDArray, eps: float = 1e-5) -> NDArray:
    # numerical Hessian for condition number analysis
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = eps, eps
            H[i, j] = (
                func(x + ei + ej) - func(x + ei - ej)
                - func(x - ei + ej) + func(x - ei - ej)
            ) / (4 * eps ** 2)
    return H


def condition_number(func: Callable, x: NDArray) -> float:
    # condition number of the Hessian at point x
    H = hessian_numerical(func, x)
    eigvals = np.linalg.eigvalsh(H)
    eigvals = eigvals[np.abs(eigvals) > 1e-12]
    if len(eigvals) == 0:
        return np.inf
    return float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))


def make_counted(
    func: Callable, grad: Callable
) -> tuple[Callable, Callable, dict[str, int]]:
    # wrapper that counts function and gradient evaluations
    counters = {"func": 0, "grad": 0}

    def counted_func(x):
        counters["func"] += 1
        return func(x)

    def counted_grad(x):
        counters["grad"] += 1
        return grad(x)

    return counted_func, counted_grad, counters
