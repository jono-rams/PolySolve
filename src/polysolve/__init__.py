import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import warnings

# Attempt to import CuPy for CUDA acceleration.
# If CuPy is not installed, the CUDA functionality will not be available.
try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# The CUDA kernel for the fitness function
_FITNESS_KERNEL = """
extern "C" __global__ void fitness_kernel(
    const long long* coefficients, 
    int num_coefficients, 
    const double* x_vals, 
    double* ranks, 
    int size, 
    double y_val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        double ans = 0;
        int lrgst_expo = num_coefficients - 1;
        for (int i = 0; i < num_coefficients; ++i)
        {
            ans += coefficients[i] * pow(x_vals[idx], (double)(lrgst_expo - i));
        }

        ans -= y_val;
        ranks[idx] = (ans == 0) ? 1.7976931348623157e+308 : fabs(1.0 / ans);
    }
}
"""

@dataclass
class GA_Options:
    """
    Configuration options for the genetic algorithm used to find function roots.

    Attributes:
        min_range (float): The minimum value for the initial random solutions.
        max_range (float): The maximum value for the initial random solutions.
        num_of_generations (int): The number of iterations the algorithm will run.
        sample_size (int): The number of top solutions to keep and return.
        data_size (int): The total number of solutions generated in each generation.
        mutation_percentage (float): The amount by which top solutions are mutated each generation.
    """
    min_range: float = -100.0
    max_range: float = 100.0
    num_of_generations: int = 10
    sample_size: int = 1000
    data_size: int = 100000
    mutation_percentage: float = 0.01

class Function:
    """
    Represents an exponential function (polynomial) of the form:
    c_0*x^n + c_1*x^(n-1) + ... + c_n
    """
    def __init__(self, largest_exponent: int):
        """
        Initializes a function with its highest degree.

        Args:
            largest_exponent (int): The largest exponent (n) in the function.
        """
        if not isinstance(largest_exponent, int) or largest_exponent < 0:
            raise ValueError("largest_exponent must be a non-negative integer.")
        self._largest_exponent = largest_exponent
        self.coefficients: Optional[np.ndarray] = None
        self._initialized = False

    def set_coeffs(self, coefficients: List[int]):
        """
        Sets the coefficients of the polynomial.

        Args:
            coefficients (List[int]): A list of integer coefficients. The list size
                                   must be largest_exponent + 1.

        Raises:
            ValueError: If the input is invalid.
        """
        expected_size = self._largest_exponent + 1
        if len(coefficients) != expected_size:
            raise ValueError(
                f"Function with exponent {self._largest_exponent} requires {expected_size} coefficients, "
                f"but {len(coefficients)} were given."
            )
        if coefficients[0] == 0 and self._largest_exponent > 0:
            raise ValueError("The first constant (for the largest exponent) cannot be 0.")

        self.coefficients = np.array(coefficients, dtype=np.int64)
        self._initialized = True

    def _check_initialized(self):
        """Raises a RuntimeError if the function coefficients have not been set."""
        if not self._initialized:
            raise RuntimeError("Function is not fully initialized. Call .set_coeffs() first.")

    @property
    def largest_exponent(self) -> int:
        """Returns the largest exponent of the function."""
        return self._largest_exponent

    def solve_y(self, x_val: float) -> float:
        """
        Solves for y given an x value. (i.e., evaluates the polynomial at x).

        Args:
            x_val (float): The x-value to evaluate.

        Returns:
            float: The resulting y-value.
        """
        self._check_initialized()
        return np.polyval(self.coefficients, x_val)

    def differential(self) -> 'Function':
        """
        Calculates the derivative of the function.

        Returns:
            Function: A new Function object representing the derivative.
        """
        self._check_initialized()
        if self._largest_exponent == 0:
            raise ValueError("Cannot differentiate a constant (Function of degree 0).")

        derivative_coefficients = np.polyder(self.coefficients)
        
        diff_func = Function(self._largest_exponent - 1)
        diff_func.set_coeffs(derivative_coefficients.tolist())
        return diff_func

    def get_real_roots(self, options: GA_Options = GA_Options(), use_cuda: bool = False) -> np.ndarray:
        """
        Uses a genetic algorithm to find the approximate real roots of the function (where y=0).

        Args:
            options (GA_Options): Configuration for the genetic algorithm.
            use_cuda (bool): If True, attempts to use CUDA for acceleration.

        Returns:
            np.ndarray: An array of approximate root values.
        """
        self._check_initialized()
        return self.solve_x(0.0, options, use_cuda)

    def solve_x(self, y_val: float, options: GA_Options = GA_Options(), use_cuda: bool = False) -> np.ndarray:
        """
        Uses a genetic algorithm to find x-values for a given y-value.

        Args:
            y_val (float): The target y-value.
            options (GA_Options): Configuration for the genetic algorithm.
            use_cuda (bool): If True, attempts to use CUDA for acceleration.

        Returns:
            np.ndarray: An array of approximate x-values.
        """
        self._check_initialized()
        if use_cuda and _CUPY_AVAILABLE:
            return self._solve_x_cuda(y_val, options)
        else:
            if use_cuda:
                warnings.warn(
                    "use_cuda=True was specified, but CuPy is not installed. "
                    "Falling back to NumPy (CPU). For GPU acceleration, "
                    "install with 'pip install polysolve[cuda]'.",
                    UserWarning
                )
    
            return self._solve_x_numpy(y_val, options)

    def _solve_x_numpy(self, y_val: float, options: GA_Options) -> np.ndarray:
        """Genetic algorithm implementation using NumPy (CPU)."""
        # Create initial random solutions
        solutions = np.random.uniform(options.min_range, options.max_range, options.data_size)

        for _ in range(options.num_of_generations):
            # Calculate fitness for all solutions (vectorized)
            y_calculated = np.polyval(self.coefficients, solutions)
            error = y_calculated - y_val
            
            ranks = np.where(error == 0, np.finfo(float).max, np.abs(1.0 / error))
            
            # Sort solutions by fitness (descending)
            sorted_indices = np.argsort(-ranks)
            solutions = solutions[sorted_indices]

            # Keep only the top solutions
            top_solutions = solutions[:options.sample_size]
            
            # For the next generation, start with the mutated top solutions
            # and fill the rest with new random values.
            mutation_factors = np.random.uniform(
                1 - options.mutation_percentage,
                1 + options.mutation_percentage,
                options.sample_size
            )
            mutated_solutions = top_solutions * mutation_factors
            
            new_random_solutions = np.random.uniform(
                options.min_range, options.max_range, options.data_size - options.sample_size
            )
            
            solutions = np.concatenate([mutated_solutions, new_random_solutions])

        # Final sort of the best solutions from the last generation
        final_solutions = np.sort(solutions[:options.sample_size])
        return final_solutions

    def _solve_x_cuda(self, y_val: float, options: GA_Options) -> np.ndarray:
        """Genetic algorithm implementation using CuPy (GPU/CUDA)."""
        # Load the raw CUDA kernel
        fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL, 'fitness_kernel')
        
        # Move coefficients to GPU
        d_coefficients = cupy.array(self.coefficients, dtype=cupy.int64)
        
        # Create initial random solutions on the GPU
        d_solutions = cupy.random.uniform(
            options.min_range, options.max_range, options.data_size, dtype=cupy.float64
        )
        d_ranks = cupy.empty(options.data_size, dtype=cupy.float64)

        # Configure kernel launch parameters
        threads_per_block = 512
        blocks_per_grid = (options.data_size + threads_per_block - 1) // threads_per_block

        for i in range(options.num_of_generations):
            # Run the fitness kernel on the GPU
            fitness_gpu(
                (blocks_per_grid,), (threads_per_block,),
                (d_coefficients, d_coefficients.size, d_solutions, d_ranks, d_solutions.size, y_val)
            )
            
            # Sort solutions by rank on the GPU
            sorted_indices = cupy.argsort(-d_ranks)
            d_solutions = d_solutions[sorted_indices]
            
            if i + 1 == options.num_of_generations:
                break

            # Get top solutions
            d_top_solutions = d_solutions[:options.sample_size]

            # Mutate top solutions on the GPU
            mutation_factors = cupy.random.uniform(
                1 - options.mutation_percentage, 1 + options.mutation_percentage, options.sample_size
            )
            d_mutated = d_top_solutions * mutation_factors
            
            # Create new random solutions for the rest
            d_new_random = cupy.random.uniform(
                options.min_range, options.max_range, options.data_size - options.sample_size
            )

            d_solutions = cupy.concatenate([d_mutated, d_new_random])

        # Get the final sample, sort it, and copy back to CPU
        final_solutions_gpu = cupy.sort(d_solutions[:options.sample_size])
        return final_solutions_gpu.get()


    def __str__(self) -> str:
        """Returns a human-readable string representation of the function."""
        self._check_initialized()
        parts = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue

            power = self._largest_exponent - i
            
            # Coefficient part
            if c == 1 and power != 0:
                coeff = ""
            elif c == -1 and power != 0:
                coeff = "-"
            else:
                coeff = str(c)

            # Variable part
            if power == 0:
                var = ""
            elif power == 1:
                var = "x"
            else:
                var = f"x^{power}"

            # Add sign for non-leading terms
            sign = ""
            if i > 0:
                sign = " + " if c > 0 else " - "
                coeff = str(abs(c))
                if abs(c) == 1 and power != 0:
                    coeff = "" # Don't show 1 for non-constant terms

            parts.append(f"{sign}{coeff}{var}")
        
        # Join parts and clean up
        result = "".join(parts)
        if result.startswith(" + "):
            result = result[3:]
        return result if result else "0"

    def __repr__(self) -> str:
        return f"Function(str='{self}')"

    def __add__(self, other: 'Function') -> 'Function':
        """Adds two Function objects."""
        self._check_initialized()
        other._check_initialized()

        new_coefficients = np.polyadd(self.coefficients, other.coefficients)
        
        result_func = Function(len(new_coefficients) - 1)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func

    def __sub__(self, other: 'Function') -> 'Function':
        """Subtracts another Function object from this one."""
        self._check_initialized()
        other._check_initialized()

        new_coefficients = np.polysub(self.coefficients, other.coefficients)
        
        result_func = Function(len(new_coefficients) - 1)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func
        
    def __mul__(self, scalar: int) -> 'Function':
        """Multiplies the function by a scalar constant."""
        self._check_initialized()
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ValueError("Cannot multiply a function by 0.")

        new_coefficients = self.coefficients * scalar
        
        result_func = Function(self._largest_exponent)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func

    def __rmul__(self, scalar: int) -> 'Function':
        """Handles scalar multiplication from the right (e.g., 3 * func)."""
        return self.__mul__(scalar)
        
    def __imul__(self, scalar: int) -> 'Function':
        """Performs in-place multiplication by a scalar (func *= 3)."""
        self._check_initialized()
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ValueError("Cannot multiply a function by 0.")
        
        self.coefficients *= scalar
        return self


def quadratic_solve(f: Function) -> Optional[List[float]]:
    """
    Calculates the real roots of a quadratic function using the quadratic formula.

    Args:
        f (Function): A Function object of degree 2.

    Returns:
        Optional[List[float]]: A list containing the two real roots, or None if there are no real roots.
    """
    f._check_initialized()
    if f.largest_exponent != 2:
        raise ValueError("Input function must be quadratic (degree 2).")

    a, b, c = f.coefficients

    discriminant = (b**2) - (4*a*c)

    if discriminant < 0:
        return None  # No real roots
    
    sqrt_discriminant = math.sqrt(discriminant)
    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)

    return [root1, root2]

# Example Usage
if __name__ == '__main__':
    print("--- Demonstrating Functionality ---")

    # Create a quadratic function: 2x^2 - 3x - 5
    f1 = Function(2)
    f1.set_coeffs([2, -3, -5])
    print(f"Function f1: {f1}")

    # Solve for y
    y = f1.solve_y(5)
    print(f"Value of f1 at x=5 is: {y}") # Expected: 2*(25) - 3*(5) - 5 = 50 - 15 - 5 = 30

    # Find the derivative: 4x - 3
    df1 = f1.differential()
    print(f"Derivative of f1: {df1}")

    # --- Root Finding ---
    # 1. Analytical solution for quadratic
    roots_analytic = quadratic_solve(f1)
    print(f"Analytic roots of f1: {roots_analytic}") # Expected: -1, 2.5

    # 2. Genetic algorithm solution
    ga_opts = GA_Options(num_of_generations=20, data_size=50000, sample_size=10)
    print("\nFinding roots with Genetic Algorithm (CPU)...")
    roots_ga_cpu = f1.get_real_roots(ga_opts)
    print(f"Approximate roots from GA (CPU): {roots_ga_cpu}")
    print("(Note: GA provides approximations around the true roots)")

    # 3. CUDA accelerated genetic algorithm
    if _CUPY_AVAILABLE:
        print("\nFinding roots with Genetic Algorithm (CUDA)...")
        # Since this PC has an RTX 4060 Ti, we can use the CUDA version.
        roots_ga_gpu = f1.get_real_roots(ga_opts, use_cuda=True)
        print(f"Approximate roots from GA (GPU): {roots_ga_gpu}")
    else:
        print("\nSkipping CUDA example: CuPy library not found or no compatible GPU.")

    # --- Function Arithmetic ---
    print("\n--- Function Arithmetic ---")
    f2 = Function(1)
    f2.set_coeffs([1, 10]) # x + 10
    print(f"Function f2: {f2}")

    # Addition: (2x^2 - 3x - 5) + (x + 10) = 2x^2 - 2x + 5
    f_add = f1 + f2
    print(f"f1 + f2 = {f_add}")

    # Subtraction: (2x^2 - 3x - 5) - (x + 10) = 2x^2 - 4x - 15
    f_sub = f1 - f2
    print(f"f1 - f2 = {f_sub}")

    # Multiplication: (x + 10) * 3 = 3x + 30
    f_mul = f2 * 3
    print(f"f2 * 3 = {f_mul}")