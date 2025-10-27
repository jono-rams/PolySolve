import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import warnings

# Attempt to import CuPy for CUDA acceleration.
# If CuPy is not installed, the CUDA functionality will not be available.
try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

# The CUDA kernels for the fitness function
_FITNESS_KERNEL_FLOAT = """
extern "C" __global__ void fitness_kernel(
    const double* coefficients, 
    int num_coefficients, 
    const double* x_vals, 
    double* ranks, 
    int size, 
    double y_val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        double ans = coefficients[0];
        for (int i = 1; i < num_coefficients; ++i)
        {
            ans = ans * x_vals[idx] + coefficients[i];
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
                           Default: -100.0
        max_range (float): The maximum value for the initial random solutions.
                           Default: 100.0
        num_of_generations (int): The number of iterations the algorithm will run.
                                  Default: 10
        sample_size (int): The number of top solutions to *return* at the end.
                           Default: 1000
        data_size (int): The total number of solutions (population size)
                         generated in each generation. Default: 100000
        mutation_strength (float): The percentage (e.g., 0.01 for 1%) by which
                                   a solution is mutated. Default: 0.01
        elite_ratio (float): The percentage (e.g., 0.05 for 5%) of the *best*
                             solutions to carry over to the next generation
                             unchanged (elitism). Default: 0.05
        crossover_ratio (float): The percentage (e.g., 0.45 for 45%) of the next
                                 generation to be created by "breeding" two
                                 solutions from the parent pool. Default: 0.45
        mutation_ratio (float): The percentage (e.g., 0.40 for 40%) of the next
                                generation to be created by mutating solutions
                                from the parent pool. Default: 0.40
    """
    min_range: float = -100.0
    max_range: float = 100.0
    num_of_generations: int = 10
    sample_size: int = 1000
    data_size: int = 100000
    mutation_strength: float = 0.01
    elite_ratio: float = 0.05
    crossover_ratio: float = 0.45
    mutation_ratio: float = 0.40

    def __post_init__(self):
        """Validates the GA options after initialization."""
        total_ratio = self.elite_ratio + self.crossover_ratio + self.mutation_ratio
        if total_ratio > 1.0:
            raise ValueError(
                f"The sum of elite_ratio, crossover_ratio, and mutation_ratio must be <= 1.0, but got {total_ratio}"
            )
        if any(r < 0 for r in [self.elite_ratio, self.crossover_ratio, self.mutation_ratio]):
            raise ValueError("GA ratios cannot be negative.")
        if self.data_size < self.sample_size:
            warnings.warn(
                f"data_size ({self.data_size}) is less than sample_size ({self.sample_size}). "
                "The number of returned solutions will be limited to data_size."
            )

def _get_cauchy_bound(coeffs: np.ndarray) -> float:
    """
    Calculates Cauchy's bound for the roots of a polynomial.
    This provides a radius R such that all roots (real and complex)
    have an absolute value less than or equal to R.
    
    R = 1 + max(|c_n-1/c_n|, |c_n-2/c_n|, ..., |c_0/c_n|)
    Where c_n is the leading coefficient (coeffs[0]).
    """
    # Normalize all coefficients by the leading coefficient
    normalized_coeffs = np.abs(coeffs[1:] / coeffs[0])
    
    # The bound is 1 + the maximum of these normalized values
    R = 1 + np.max(normalized_coeffs)
    
    return R

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

    def set_coeffs(self, coefficients: List[Union[int, float]]):
        """
        Sets the coefficients of the polynomial.

        Args:
            coefficients (List[Union[int, float]]): A list of integer or float
                                                   coefficients. The list size
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
        
        # Check if any coefficient is a float
        is_float = any(isinstance(c, float) for c in coefficients)

        # Choose the dtype based on the input
        target_dtype = np.float64 if is_float else np.int64

        self.coefficients = np.array(coefficients, dtype=target_dtype)
        self._initialized = True

    def _check_initialized(self):
        """Raises a RuntimeError if the function coefficients have not been set."""
        if not self._initialized:
            raise RuntimeError("Function is not fully initialized. Call .set_coeffs() first.")

    @property
    def largest_exponent(self) -> int:
        """Returns the largest exponent of the function."""
        return self._largest_exponent
    
    @property
    def degree(self) -> int:
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
        warnings.warn(
            "The 'differential' function has been renamed. Please use 'derivative' instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self._check_initialized()
        if self._largest_exponent == 0:
            raise ValueError("Cannot differentiate a constant (Function of degree 0).")

        return self.derivative()
        
    
    def derivative(self) -> 'Function':
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
    

    def nth_derivative(self, n: int) -> 'Function':
        """
        Calculates the nth derivative of the function.

        Args:
            n (int): The order of the derivative to calculate.

        Returns:
           Function: A new Function object representing the nth derivative.
        """
        self._check_initialized()
        
        if not isinstance(n, int) or n < 1:
            raise ValueError("Derivative order 'n' must be a positive integer.")

        if n > self.largest_exponent:
            function = Function(0)
            function.set_coeffs([0])
            return function

        if n == 1:
            return self.derivative()
        
        function = self
        for _ in range(n):
            function = function.derivative()

        return function


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
        elite_ratio = options.elite_ratio
        crossover_ratio = options.crossover_ratio
        mutation_ratio = options.mutation_ratio
        
        data_size = options.data_size
        
        elite_size = int(data_size * elite_ratio)
        crossover_size = int(data_size * crossover_ratio)
        mutation_size = int(data_size * mutation_ratio)
        random_size = data_size - elite_size - crossover_size - mutation_size

        # Check if the user is using the default, non-expert range
        user_range_is_default = (options.min_range == -100.0 and options.max_range == 100.0)

        if user_range_is_default:
            # User hasn't specified a custom range.
            # We are the expert; use the smart, guaranteed bound.
            bound = _get_cauchy_bound(self.coefficients)
            min_r = -bound
            max_r = bound
        else:
            # User has provided a custom range.
            # Trust the expert; use their range.
            min_r = options.min_range
            max_r = options.max_range

        # Create initial random solutions
        solutions = np.random.uniform(min_r, max_r, data_size)

        for _ in range(options.num_of_generations):
            # Calculate fitness for all solutions (vectorized)
            y_calculated = np.polyval(self.coefficients, solutions)
            error = y_calculated - y_val
            
            with np.errstate(divide='ignore'):
                ranks = np.where(error == 0, np.finfo(float).max, np.abs(1.0 / error))
            
            # Sort solutions by fitness (descending)
            sorted_indices = np.argsort(-ranks)
            solutions = solutions[sorted_indices]

            # --- Create the next generation ---

            # 1. Elitism: Keep the best solutions as-is
            elite_solutions = solutions[:elite_size]

            # Define a "parent pool" of the top 50% of solutions to breed from
            parent_pool = solutions[:data_size // 2]

            # 2. Crossover: Breed two parents to create a child
            # Select from the full list (indices 0 to data_size-1)
            parent1_indices = np.random.randint(0, data_size, crossover_size)
            parent2_indices = np.random.randint(0, data_size, crossover_size)
            parents1 = solutions[parent1_indices]
            parents2 = solutions[parent2_indices]
            # Simple "average" crossover
            crossover_solutions = (parents1 + parents2) / 2.0

            # 3. Mutation:
            # Select from the full list (indices 0 to data_size-1)
            mutation_candidates = solutions[np.random.randint(0, data_size, mutation_size)]
            
            # Use mutation_strength (the new name)
            mutation_factors = np.random.uniform(
                1 - options.mutation_strength,
                1 + options.mutation_strength,
                mutation_size
            )
            mutated_solutions = mutation_candidates * mutation_factors

            # 4. New Randoms: Add new blood to prevent getting stuck
            random_solutions = np.random.uniform(min_r, max_r, random_size)
            
            # Assemble the new generation
            solutions = np.concatenate([
                elite_solutions, 
                crossover_solutions, 
                mutated_solutions, 
                random_solutions
            ])

        # --- Final Step: Return the best results ---
        # After all generations, do one last ranking to find the best solutions
        y_calculated = np.polyval(self.coefficients, solutions)
        error = y_calculated - y_val
        with np.errstate(divide='ignore'):
            ranks = np.where(error == 0, np.finfo(float).max, np.abs(1.0 / error))
        sorted_indices = np.argsort(-ranks)
        
        # Get the top 'sample_size' solutions the user asked for
        best_solutions = solutions[sorted_indices][:options.sample_size]
        
        return np.sort(best_solutions)

    def _solve_x_cuda(self, y_val: float, options: GA_Options) -> np.ndarray:
        """Genetic algorithm implementation using CuPy (GPU/CUDA)."""

        elite_ratio = options.elite_ratio
        crossover_ratio = options.crossover_ratio
        mutation_ratio = options.mutation_ratio
        
        data_size = options.data_size
        
        elite_size = int(data_size * elite_ratio)
        crossover_size = int(data_size * crossover_ratio)
        mutation_size = int(data_size * mutation_ratio)
        random_size = data_size - elite_size - crossover_size - mutation_size
        
        # ALWAYS cast coefficients to float64 for the kernel.
        fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL_FLOAT, 'fitness_kernel')
        d_coefficients = cupy.array(self.coefficients, dtype=cupy.float64)
        
        # Check if the user is using the default, non-expert range
        user_range_is_default = (options.min_range == -100.0 and options.max_range == 100.0)

        if user_range_is_default:
            # User hasn't specified a custom range.
            # We are the expert; use the smart, guaranteed bound.
            bound = _get_cauchy_bound(self.coefficients)
            min_r = -bound
            max_r = bound
        else:
            # User has provided a custom range.
            # Trust the expert; use their range.
            min_r = options.min_range
            max_r = options.max_range

        # Create initial random solutions on the GPU
        d_solutions = cupy.random.uniform(
            min_r, max_r, options.data_size, dtype=cupy.float64
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
            
            # --- Create the next generation ---
            
            # 1. Elitism
            d_elite_solutions = d_solutions[:elite_size]

            # Define a "parent pool" of the top 50% of solutions to breed from
            d_parent_pool = d_solutions[:data_size // 2]
            parent_pool_size = d_parent_pool.size

            # 2. Crossover
            # Select from the full list (indices 0 to data_size-1)
            parent1_indices = cupy.random.randint(0, data_size, crossover_size)
            parent2_indices = cupy.random.randint(0, data_size, crossover_size)
            d_parents1 = d_solutions[parent1_indices]
            d_parents2 = d_solutions[parent2_indices]
            d_crossover_solutions = (d_parents1 + d_parents2) / 2.0

            # 3. Mutation
            # Select from the full list (indices 0 to data_size-1)
            mutation_indices = cupy.random.randint(0, data_size, mutation_size)
            d_mutation_candidates = d_solutions[mutation_indices]
            
            # Use mutation_strength (the new name)
            d_mutation_factors = cupy.random.uniform(
                1 - options.mutation_strength, 
                1 + options.mutation_strength, 
                mutation_size
            )
            d_mutated_solutions = d_mutation_candidates * d_mutation_factors

            # 4. New Randoms
            d_random_solutions = cupy.random.uniform(
                min_r, max_r, random_size, dtype=cupy.float64
            )

            # Assemble the new generation
            d_solutions = cupy.concatenate([
                d_elite_solutions,
                d_crossover_solutions,
                d_mutated_solutions,
                d_random_solutions
            ])

        # --- Final Step: Return the best results ---
        # After all generations, do one last ranking to find the best solutions
        fitness_gpu(
            (blocks_per_grid,), (threads_per_block,),
            (d_coefficients, d_coefficients.size, d_solutions, d_ranks, d_solutions.size, y_val)
        )
        sorted_indices = cupy.argsort(-d_ranks)
        
        # Get the top 'sample_size' solutions
        d_best_solutions = d_solutions[sorted_indices][:options.sample_size]

        # Get the final sample, sort it, and copy back to CPU
        final_solutions_gpu = cupy.sort(d_best_solutions)
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
            coeff_val = c
            if c == int(c):
                coeff_val = int(c)

            if coeff_val == 1 and power != 0:
                coeff = ""
            elif coeff_val == -1 and power != 0:
                coeff = "-"
            else:
                coeff = str(coeff_val)

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
                coeff = str(abs(coeff_val))
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
    
    def _multiply_by_scalar(self, scalar: Union[int, float]) -> 'Function':
        """Helper method to multiply the function by a scalar constant."""
        self._check_initialized() # It's good practice to check here too

        if scalar == 0:
            result_func = Function(0)
            result_func.set_coeffs([0])
            return result_func
    
        new_coefficients = self.coefficients * scalar
    
        result_func = Function(self._largest_exponent)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func

    def _multiply_by_function(self, other: 'Function') -> 'Function':
        """Helper method for polynomial multiplication (Function * Function)."""
        self._check_initialized()
        other._check_initialized()

        # np.polymul performs convolution of coefficients to multiply polynomials
        new_coefficients = np.polymul(self.coefficients, other.coefficients)
    
        # The degree of the resulting polynomial is derived from the new coefficients
        new_degree = len(new_coefficients) - 1
    
        result_func = Function(new_degree)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func
        
    def __mul__(self, other: Union['Function', int, float]) -> 'Function':
        """Multiplies the function by a scalar constant."""
        if isinstance(other, (int, float)):
            return self._multiply_by_scalar(other)
        elif isinstance(other, self.__class__):
            return self._multiply_by_function(other)
        else:
            return NotImplemented

    def __rmul__(self, scalar: Union[int, float]) -> 'Function':
        """Handles scalar multiplication from the right (e.g., 3 * func)."""

        return self.__mul__(scalar)
        
    def __imul__(self, other: Union['Function', int, float]) -> 'Function':
        """Performs in-place multiplication by a scalar (func *= 3)."""

        self._check_initialized()
    
        if isinstance(other, (int, float)):
            if other == 0:
                self.coefficients = np.array([0], dtype=self.coefficients.dtype)
                self._largest_exponent = 0
            else:
                self.coefficients *= other
            
        elif isinstance(other, self.__class__):
            other._check_initialized()
            self.coefficients = np.polymul(self.coefficients, other.coefficients)
            self._largest_exponent = len(self.coefficients) - 1
        
        else:
            return NotImplemented
        
        return self


    def quadratic_solve(self) -> Optional[List[float]]:
        """
        Calculates the real roots of a quadratic function using the quadratic formula.

        Args:
            f (Function): A Function object of degree 2.

        Returns:
            Optional[List[float]]: A list containing the two real roots, or None if there are no real roots.
        """
        self._check_initialized()
        if self.largest_exponent != 2:
            raise ValueError("Input function must be quadratic (degree 2) to use quadratic_solve.")

        a, b, c = self.coefficients

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
    df1 = f1.derivative()
    print(f"Derivative of f1: {df1}")

    # Find the second derivative: 4
    ddf1 = f1.nth_derivative(2)
    print(f"Second derivative of f1: {ddf1}")

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

    # f3 represents 2x^2 + 3x + 1
    f3 = Function(2)
    f3.set_coeffs([2, 3, 1]) 
    print(f"Function f3: {f3}")

    # f4 represents 5x - 4
    f4 = Function(1)
    f4.set_coeffs([5, -4])
    print(f"Function f4: {f4}")

    # Multiply the two functions
    product_func = f3 * f4
    print(f"f3 * f4 = {product_func}")
