import pytest
import numpy as np

# Try to import cupy to check for CUDA availability
try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from polysolve import Function, GA_Options, quadratic_solve

@pytest.fixture
def quadratic_func() -> Function:
    """Provides a standard quadratic function: 2x^2 - 3x - 5."""
    f = Function(largest_exponent=2)
    f.set_constants([2, -3, -5])
    return f

@pytest.fixture
def linear_func() -> Function:
    """Provides a standard linear function: x + 10."""
    f = Function(largest_exponent=1)
    f.set_constants([1, 10])
    return f

# --- Core Functionality Tests ---

def test_solve_y(quadratic_func):
    """Tests if the function correctly evaluates y for a given x."""
    assert quadratic_func.solve_y(5) == 30.0
    assert quadratic_func.solve_y(0) == -5.0
    assert quadratic_func.solve_y(-1) == 0.0

def test_differential(quadratic_func):
    """Tests the calculation of the function's derivative."""
    derivative = quadratic_func.differential()
    assert derivative.largest_exponent == 1
    # The derivative of 2x^2 - 3x - 5 is 4x - 3
    assert np.array_equal(derivative.constants, [4, -3])

def test_quadratic_solve(quadratic_func):
    """Tests the analytical quadratic solver for exact roots."""
    roots = quadratic_solve(quadratic_func)
    # Sorting ensures consistent order for comparison
    assert sorted(roots) == [-1.0, 2.5]

# --- Arithmetic Operation Tests ---

def test_addition(quadratic_func, linear_func):
    """Tests the addition of two Function objects."""
    # (2x^2 - 3x - 5) + (x + 10) = 2x^2 - 2x + 5
    result = quadratic_func + linear_func
    assert result.largest_exponent == 2
    assert np.array_equal(result.constants, [2, -2, 5])

def test_subtraction(quadratic_func, linear_func):
    """Tests the subtraction of two Function objects."""
    # (2x^2 - 3x - 5) - (x + 10) = 2x^2 - 4x - 15
    result = quadratic_func - linear_func
    assert result.largest_exponent == 2
    assert np.array_equal(result.constants, [2, -4, -15])

def test_multiplication(linear_func):
    """Tests the multiplication of a Function object by a scalar."""
    # (x + 10) * 3 = 3x + 30
    result = linear_func * 3
    assert result.largest_exponent == 1
    assert np.array_equal(result.constants, [3, 30])

# --- Genetic Algorithm Root-Finding Tests ---

def test_get_real_roots_numpy(quadratic_func):
    """
    Tests that the NumPy-based genetic algorithm approximates the roots correctly.
    """
    # Using more generations for higher accuracy in testing
    ga_opts = GA_Options(num_of_generations=25, data_size=50000)
    
    roots = quadratic_func.get_real_roots(ga_opts, use_cuda=False)
    
    # Check if the algorithm found values close to the two known roots.
    # We don't know which order they'll be in, so we check for presence.
    expected_roots = np.array([-1.0, 2.5])
    
    # Check that at least one found root is close to -1.0
    assert np.any(np.isclose(roots, expected_roots[0], atol=1e-2))
    
    # Check that at least one found root is close to 2.5
    assert np.any(np.isclose(roots, expected_roots[1], atol=1e-2))


@pytest.mark.skipif(not _CUPY_AVAILABLE, reason="CuPy is not installed, skipping CUDA test.")
def test_get_real_roots_cuda(quadratic_func):
    """
    Tests that the CUDA-based genetic algorithm approximates the roots correctly.
    This test implicitly verifies that the CUDA kernel is functioning.
    It will be skipped automatically if CuPy is not available.
    """
    
    ga_opts = GA_Options(num_of_generations=25, data_size=50000)
    
    roots = quadratic_func.get_real_roots(ga_opts, use_cuda=True)
    
    expected_roots = np.array([-1.0, 2.5])
    
    # Verify that the CUDA implementation also finds the correct roots within tolerance.
    assert np.any(np.isclose(roots, expected_roots[0], atol=1e-2))
    assert np.any(np.isclose(roots, expected_roots[1], atol=1e-2))

