# polysolve

[![PyPI version](https://img.shields.io/pypi/v/polysolve.svg)](https://pypi.org/project/polysolve/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/polysolve.svg)](https://pypi.org/project/polysolve/)

A Python library for representing, manipulating, and solving polynomial equations using a high-performance genetic algorithm, with optional CUDA/GPU acceleration.

---

## Key Features

* **Create and Manipulate Polynomials**: Easily define polynomials of any degree and perform arithmetic operations like addition, subtraction, and scaling.
* **Genetic Algorithm Solver**: Find approximate real roots for complex polynomials where analytical solutions are difficult or impossible.
* **CUDA Accelerated**: Leverage NVIDIA GPUs for a massive performance boost when finding roots in large solution spaces.
* **Analytical Solvers**: Includes standard, exact solvers for simple cases (e.g., `quadratic_solve`).
* **Simple API**: Designed to be intuitive and easy to integrate into any project.

---

## Installation

Install the base package from PyPI:

```bash
pip install polysolve
```

### CUDA Acceleration

To enable GPU acceleration, install the extra that matches your installed NVIDIA CUDA Toolkit version. This provides a significant speedup for the genetic algorithm.

**For CUDA 12.x users:**
```bash
pip install polysolve[cuda12]
```

---

## Quick Start

Here is a simple example of how to define a quadratic function, find its properties, and solve for its roots.

```python
from polysolve import Function, GA_Options, quadratic_solve

# 1. Define the function f(x) = 2x^2 - 3x - 5
f1 = Function(largest_exponent=2)
f1.set_constants([2, -3, -5])

print(f"Function f1: {f1}")
# > Function f1: 2x^2 - 3x - 5

# 2. Solve for y at a given x
y_val = f1.solve_y(5)
print(f"Value of f1 at x=5 is: {y_val}")
# > Value of f1 at x=5 is: 30.0

# 3. Get the derivative: 4x - 3
df1 = f1.differential()
print(f"Derivative of f1: {df1}")
# > Derivative of f1: 4x - 3

# 4. Find roots analytically using the quadratic formula
#    This is exact and fast for degree-2 polynomials.
roots_analytic = quadratic_solve(f1)
print(f"Analytic roots: {sorted(roots_analytic)}")
# > Analytic roots: [-1.0, 2.5]

# 5. Find roots with the genetic algorithm (CPU)
#    This can solve polynomials of any degree.
ga_opts = GA_Options(num_of_generations=20)
roots_ga = f1.get_real_roots(ga_opts, use_cuda=False)
print(f"Approximate roots from GA: {roots_ga[:2]}")
# > Approximate roots from GA: [-1.000..., 2.500...]

# If you installed a CUDA extra, you can run it on the GPU:
# roots_ga_gpu = f1.get_real_roots(ga_opts, use_cuda=True)
# print(f"Approximate roots from GA (GPU): {roots_ga_gpu[:2]}")

```

---

## Development & Testing Environment

This project is automatically tested against a specific set of dependencies to ensure stability. Our Continuous Integration (CI) pipeline runs on an environment using **CUDA 12.5** on **Ubuntu 24.04**.

While the code may work on other configurations, all contributions must pass the automated tests in our reference environment. For detailed information on how to replicate the testing environment, please see our [**Contributing Guide**](CONTRIBUTING.md).

## Contributing

Contributions are welcome! Whether it's a bug report, a feature request, or a pull request, please feel free to get involved.

Please read our `CONTRIBUTING.md` file for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
