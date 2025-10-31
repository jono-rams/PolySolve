import math
import numpy as np
import numba
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

@numba.jit(nopython=True, fastmath=True, parallel=True)
def _calculate_ranks_numba(solutions, coefficients, y_val, ranks):
    """
    A Numba-jitted, parallel function to calculate fitness.
    This replaces np.polyval and the rank calculation.
    """
    num_coefficients = coefficients.shape[0]
    data_size = solutions.shape[0]
        
    # This prange will be run in parallel on all your CPU cores
    for idx in numba.prange(data_size):
        x_val = solutions[idx]
            
        # Horner's method (same as np.polyval)
        ans = coefficients[0]
        for i in range(1, num_coefficients):
            ans = ans * x_val + coefficients[i]
            
        ans -= y_val
            
        if ans == 0.0:
            ranks[idx] = 1.7976931348623157e+308 # np.finfo(float).max
        else:
            ranks[idx] = abs(1.0 / ans)

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
        selection_percentile (float): The top percentage (e.g., 0.66 for 66%)
                                      of solutions to use as the parent pool
                                      for crossover. A smaller value speeds
                                      up single-root convergence; a larger
                                      value helps find multiple roots.
                                      Default: 0.66
        blend_alpha (float): The expansion factor for Blend Crossover (BLX-alpha).
                             0.0 = average crossover (no expansion).
                             0.5 = 50% expansion beyond the parent range.
                             Default: 0.5
        root_precision (int): The number of decimal places to round roots to
                              when clustering. A smaller number (e.g., 3)
                              groups roots more aggressively. A larger number
                              (e.g., 7) is more precise but may return
                              multiple near-identical roots. Default: 5
    """
    min_range: float = -100.0
    max_range: float = 100.0
    num_of_generations: int = 10
    data_size: int = 100000
    mutation_strength: float = 0.01
    elite_ratio: float = 0.05
    crossover_ratio: float = 0.45
    mutation_ratio: float = 0.40
    selection_percentile: float = 0.66
    blend_alpha: float = 0.5
    root_precision: int = 5

    def __post_init__(self):
        """Validates the GA options after initialization."""
        total_ratio = self.elite_ratio + self.crossover_ratio + self.mutation_ratio
        if total_ratio > 1.0:
            raise ValueError(
                f"The sum of elite_ratio, crossover_ratio, and mutation_ratio must be <= 1.0, but got {total_ratio}"
            )
        if any(r < 0 for r in [self.elite_ratio, self.crossover_ratio, self.mutation_ratio]):
            raise ValueError("GA ratios cannot be negative.")
        if not (0 < self.selection_percentile <= 1.0):
            raise ValueError(
                f"selection_percentile must be between 0 (exclusive) and 1.0 (inclusive), but got {self.selection_percentile}"
            )
        if self.blend_alpha < 0:
            raise ValueError(
                f"blend_alpha cannot be negative, but got {self.blend_alpha}"
            )
        if self.root_precision > 15:
            warnings.warn(
                f"root_precision={self.root_precision} is greater than 15. "
                "This demands an accuracy that is likely impossible for standard "
                "64-bit floats (float64), which are limited to 15-16 significant digits. "
                "The solver may fail to find any roots.",
                UserWarning,
                stacklevel=2
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

        # Pre-allocate ranks array
        ranks = np.empty(data_size, dtype=np.float64)

        for _ in range(options.num_of_generations):
            # Calculate fitness for all solutions (vectorized)
            _calculate_ranks_numba(solutions, self.coefficients, y_val, ranks)

            parent_pool_size = int(data_size * options.selection_percentile)

            # 1. Get indices for the elite solutions (O(N) operation)
            #    We find the 'elite_size'-th largest element.
            elite_indices = np.argpartition(-ranks, elite_size)[:elite_size]
            
            # 2. Get indices for the parent pool (O(N) operation)
            #    We find the 'parent_pool_size'-th largest element.
            parent_pool_indices = np.argpartition(-ranks, parent_pool_size)[:parent_pool_size]

            # --- Create the next generation ---

            # 1. Elitism: Keep the best solutions as-is
            elite_solutions = solutions[elite_indices]

            # 2. Crossover: Breed two parents to create a child
            # Select from the fitter PARENT POOL
            parents1_idx = np.random.choice(parent_pool_indices, crossover_size)
            parents2_idx = np.random.choice(parent_pool_indices, crossover_size)
            
            parents1 = solutions[parents1_idx]
            parents2 = solutions[parents2_idx]
            # Blend Crossover (BLX-alpha)
            alpha = options.blend_alpha

            # Find min/max for all parent pairs
            p_min = np.minimum(parents1, parents2)
            p_max = np.maximum(parents1, parents2)

            # Calculate range (I)
            parent_range = p_max - p_min

            # Calculate new min/max for the expanded range
            new_min = p_min - (alpha * parent_range)
            new_max = p_max + (alpha * parent_range)

            # Create a new random child within the expanded range
            crossover_solutions = np.random.uniform(new_min, new_max, crossover_size)

            # 3. Mutation:
            # Select from the full list (indices 0 to data_size-1)
            mutation_candidates = solutions[np.random.randint(0, data_size, mutation_size)]
            
            # Use mutation_strength
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
        _calculate_ranks_numba(solutions, self.coefficients, y_val, ranks)
        
        # 1. Define quality based on the user's desired precision
        #    (e.g., precision=5 -> rank > 1e6, precision=8 -> rank > 1e9)
        #    We add +1 for a buffer, ensuring we only get high-quality roots.
        quality_threshold = 10**(options.root_precision + 1)

        # 2. Get all solutions that meet this quality threshold
        high_quality_solutions = solutions[ranks > quality_threshold]

        if high_quality_solutions.size == 0:
            # No roots found that meet the quality, return empty
            return np.array([])
        
        # 3. Cluster these high-quality solutions by rounding
        rounded_solutions = np.round(high_quality_solutions, options.root_precision)

        # 4. Return only the unique roots
        unique_roots = np.unique(rounded_solutions)
        
        return np.sort(unique_roots)

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

            # 2. Crossover
            parent_pool_size = int(data_size * options.selection_percentile)
            # Select from the fitter PARENT POOL
            parent1_indices = cupy.random.randint(0, parent_pool_size, crossover_size)
            parent2_indices = cupy.random.randint(0, parent_pool_size, crossover_size)
            # Get parents directly from the sorted solutions array using the pool-sized indices
            d_parents1 = d_solutions[parent1_indices]
            d_parents2 = d_solutions[parent2_indices]
            
            # Blend Crossover (BLX-alpha)
            alpha = options.blend_alpha

            # Find min/max for all parent pairs
            d_p_min = cupy.minimum(d_parents1, d_parents2)
            d_p_max = cupy.maximum(d_parents1, d_parents2)

            # Calculate range (I)
            d_parent_range = d_p_max - d_p_min

            # Calculate new min/max for the expanded range
            d_new_min = d_p_min - (alpha * d_parent_range)
            d_new_max = d_p_max + (alpha * d_parent_range)

            # Create a new random child within the expanded range
            d_crossover_solutions = cupy.random.uniform(d_new_min, d_new_max, crossover_size)

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
        
        # 1. Define quality based on the user's desired precision
        #    (e.g., precision=5 -> rank > 1e6, precision=8 -> rank > 1e9)
        #    We add +1 for a buffer, ensuring we only get high-quality roots.
        quality_threshold = 10**(options.root_precision + 1)
        
        # 2. Get all solutions that meet this quality threshold
        d_high_quality_solutions = d_solutions[d_ranks > quality_threshold]

        if d_high_quality_solutions.size == 0:
            return np.array([])
            
        # 3. Cluster these high-quality solutions on the GPU by rounding
        d_rounded_solutions = cupy.round(d_high_quality_solutions, options.root_precision)
        
        # 4. Get only the unique roots
        d_unique_roots = cupy.unique(d_rounded_solutions)

        # Sort the unique roots and copy back to CPU
        final_solutions_gpu = cupy.sort(d_unique_roots)
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
    roots_analytic = f1.quadratic_solve()
    print(f"Analytic roots of f1: {roots_analytic}") # Expected: -1, 2.5

    # 2. Genetic algorithm solution
    ga_opts = GA_Options(num_of_generations=20, data_size=50000)
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
