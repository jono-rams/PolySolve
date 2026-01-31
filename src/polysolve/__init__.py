import math
import cmath
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
    const double* __restrict__ coefficients, 
    int num_coefficients, 
    const double* __restrict__ x_vals, 
    double* __restrict__ ranks, 
    int size, 
    double y_val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        double ans = coefficients[0];
        double x = x_vals[idx];

        for (int i = 1; i < num_coefficients; ++i)
        {
            ans = ans * x + coefficients[i];
        }

        ans -= y_val;
        
        if (isinf(ans) || isnan(ans)) {
            ranks[idx] = 0.0;
        } else {
            ranks[idx] = 1.0 / (fabs(ans) + 1e-15);
        }
    }
}
"""

_FITNESS_KERNEL_FLOAT_DYNAMIC = """
extern "C" __global__ void fitness_kernel_shared(
    const double* __restrict__ coefficients, 
    int num_coefficients, 
    const double* __restrict__ x_vals, 
    double* __restrict__ ranks, 
    int size, 
    double y_val)
{
    // Dynamic Shared Memory declaration
    extern __shared__ double s_coeffs[];

    for (int i = threadIdx.x; i < num_coefficients; i += blockDim.x) {
        s_coeffs[i] = coefficients[i];
    }

    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        double ans = s_coeffs[0];
        double x = x_vals[idx];

        for (int i = 1; i < num_coefficients; ++i)
        {
            ans = ans * x + s_coeffs[i];
        }

        ans -= y_val;
        
        if (isinf(ans) || isnan(ans)) {
            ranks[idx] = 0.0;
        } else {
            ranks[idx] = 1.0 / (fabs(ans) + 1e-15);
        }
    }
}
"""

_FITNESS_KERNEL_COMPLEX = """
struct Complex {
    double r;
    double i;
};

__device__ Complex c_add(Complex a, Complex b) {
    return {a.r + b.r, a.i + b.i};
}

__device__ Complex c_mul(Complex a, Complex b) {
    return {
        a.r * b.r - a.i * b.i, 
        a.r * b.i + a.i * b.r 
    };
}

__device__ double c_abs(Complex a) {
    return sqrt(a.r * a.r + a.i * a.i);
}

extern "C" __global__ void fitness_kernel_complex(
    const double* __restrict__ coeffs_real, 
    const double* __restrict__ coeffs_imag,
    int num_coefficients, 
    const double* __restrict__ sol_real,
    const double* __restrict__ sol_imag,
    double* __restrict__ ranks, 
    int size,
    double y_real,
    double y_imag)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        Complex x = {sol_real[idx], sol_imag[idx]};
        Complex ans = {coeffs_real[0], coeffs_imag[0]};

        for (int i = 1; i < num_coefficients; ++i)
        {
            Complex c = {coeffs_real[i], coeffs_imag[i]};
            ans = c_mul(ans, x);
            ans = c_add(ans, c);
        }

        Complex diff = {ans.r - y_real, ans.i - y_imag};
        
        if (isinf(diff.r) || isinf(diff.i) || isnan(diff.r) || isnan(diff.i)) {
            ranks[idx] = 0.0;
        } else {
            double modulus = hypot(diff.r, diff.i); 
            ranks[idx] = 1.0 / (modulus + 1e-15);
        }
    }
}
"""

_FITNESS_KERNEL_COMPLEX_DYNAMIC = """
struct Complex {
    double r;
    double i;
};

__device__ Complex c_add(Complex a, Complex b) {
    return {a.r + b.r, a.i + b.i};
}

__device__ Complex c_mul(Complex a, Complex b) {
    return {
        a.r * b.r - a.i * b.i, 
        a.r * b.i + a.i * b.r 
    };
}

__device__ double c_abs(Complex a) {
    return sqrt(a.r * a.r + a.i * a.i);
}

extern "C" __global__ void fitness_kernel_complex_shared(
    const double* __restrict__ coeffs_real, 
    const double* __restrict__ coeffs_imag,
    int num_coefficients, 
    const double* __restrict__ sol_real,
    const double* __restrict__ sol_imag,
    double* __restrict__ ranks, 
    int size,
    double y_real,
    double y_imag)
{
    // Dynamic Shared Memory declaration
    extern __shared__ double s_memory[];

    for (int i = threadIdx.x; i < num_coefficients; i += blockDim.x) {
        s_memory[2 * i]     = coeffs_real[i];
        s_memory[2 * i + 1] = coeffs_imag[i];
    }

    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        Complex x = {sol_real[idx], sol_imag[idx]};
        Complex ans = {s_memory[0], s_memory[1]};

        for (int i = 1; i < num_coefficients; ++i)
        {
            Complex c = {s_memory[2 * i], s_memory[2 * i + 1]};
            ans = c_mul(ans, x);
            ans = c_add(ans, c);
        }

        Complex diff = {ans.r - y_real, ans.i - y_imag};

        if (isinf(diff.r) || isinf(diff.i) || isnan(diff.r) || isnan(diff.i)) {
            ranks[idx] = 0.0;
        } else {
            double modulus = hypot(diff.r, diff.i); 
            ranks[idx] = 1.0 / (modulus + 1e-15);
        }
    }
}
"""

@numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
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
            
        ranks[idx] = 1.0 / (abs(ans) + 1e-15)


@numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _calculate_ranks_complex_numba(solutions, coefficients, y_val, ranks):
    """
    Parallel fitness calculation for Complex numbers on CPU.
    Solutions and Coefficients must be of type complex128.
    """
    num_coefficients = coefficients.shape[0]
    data_size = solutions.shape[0]

    for idx in numba.prange(data_size):
        x_val = solutions[idx]

        # Initialize with the leading coefficient
        ans = coefficients[0]

        # Horner's Method
        for i in range(1, num_coefficients):
            ans = ans * x_val + coefficients[i]

        ans -= y_val

        # Calculate rank based on Modulus (Magnitude)
        # abs(z) for a complex number returns sqrt(a^2 + b^2)
        modulus = abs(ans)

        ranks[idx] = 1.0 / (modulus + 1e-15)


@dataclass
class GA_Options:
    """
    Configuration options for the genetic algorithm used to find function roots.

    Attributes:
        min_range (float): The minimum value for the initial random solutions.
                           Default: 0.0
        max_range (float): The maximum value for the initial random solutions.
                           Default: 0.0
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
        find_complex (bool): Whether to find complex roots as well. Default: True
    """
    min_range: float = 0.0 # Returned for backwards compatibility even though it's no longer used
    max_range: float = 0.0 # Returned for backwards compatibility even though it's no longer used
    num_of_generations: int = 10
    data_size: int = 100000
    mutation_strength: float = 0.01
    elite_ratio: float = 0.05
    crossover_ratio: float = 0.45
    mutation_ratio: float = 0.40
    selection_percentile: float = 0.66
    blend_alpha: float = 0.5
    root_precision: int = 5
    find_complex: bool = True

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
        if self.min_range != 0.0 or self.max_range != 0.0:
            warnings.warn(
                "The 'min_range' and 'max_range' parameters are deprecated and will be ignored. "
                "Search bounds are now automatically calculated using Cauchy's bound.",
                DeprecationWarning,
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
    if len(coeffs) <= 1:
        return 1000.0
    
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
    def __init__(self, largest_exponent: int, coefficients: Optional[List[Union[int, float, complex]]] = None):
        """
        Initializes a function with its highest degree.

        Args:
            largest_exponent (int): The largest exponent (n) in the function.
        """
        self._largest_exponent = largest_exponent
        if coefficients is not None:
            self.set_coeffs(coefficients)
            # Verify user provided exponent matches if they provided both
            if largest_exponent is not None and self._largest_exponent != largest_exponent:
                raise ValueError("Provided largest_exponent does not match coefficient list length.")
        elif largest_exponent is not None:
            self.coefficients = None
            self._initialized = False
        else:
            raise ValueError("Must provide either coefficients or largest_exponent.")

    def set_coeffs(self, coefficients: List[Union[int, float, complex]]):
        """
        Sets the coefficients of the polynomial.

        Args:
            coefficients (List[Union[int, float]]): A list of integer, float or complex
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
        
        # Check for complex, then float, then int
        is_complex = any(isinstance(c, complex) for c in coefficients)

        # Choose the dtype based on the input
        if is_complex:
            target_dtype = np.complex128
        else:
            target_dtype = np.float64

        self.coefficients = np.array(coefficients, dtype=target_dtype)
        self._largest_exponent = len(coefficients) - 1
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
            diff_func = Function(0)
            diff_func.set_coeffs([0])
            return diff_func
        
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


    def get_real_roots(self, options: Optional[GA_Options] = None, use_cuda: bool = False) -> np.ndarray:
        """
        Uses a genetic algorithm to find the approximate real roots of the function (where y=0).

        Args:
            options (GA_Options): Configuration for the genetic algorithm.
            use_cuda (bool): If True, attempts to use CUDA for acceleration.

        Returns:
            np.ndarray: An array of approximate root values.
        """
        self._check_initialized()
        if options is None:
            options = GA_Options()
        import copy
        safe_options = copy.copy(options)
        safe_options.find_complex = False
        return self.solve_x(0.0, safe_options, use_cuda)
    

    def get_roots(self, options: Optional[GA_Options] = None, use_cuda: bool = False) -> np.ndarray:
        """
        Uses a genetic algorithm to find the approximate roots of the function (where y=0).

        Args:
            options (GA_Options): Configuration for the genetic algorithm.
            use_cuda (bool): If True, attempts to use CUDA for acceleration.

        Returns:
            np.ndarray: An array of approximate root values.
        """
        self._check_initialized()
        if options is None:
            options = GA_Options()
        return self.solve_x(0.0, options, use_cuda)


    def solve_x(self, y_val: Union[float, complex], options: Optional[GA_Options] = None, use_cuda: bool = False) -> np.ndarray:
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
        if options is None:
            options = GA_Options()
        if options.find_complex:
            target_y = complex(y_val)

            if use_cuda and _CUPY_AVAILABLE:
                return self._solve_complex_cuda(target_y, options)
            else:
                if use_cuda:
                # Warn if user wanted CUDA but it's not available
                    warnings.warn(
                        "use_cuda=True was specified, but CuPy is not installed. "
                        "Falling back to NumPy (CPU) for complex roots.",
                        UserWarning
                    )
                return self._solve_complex_numpy(target_y, options)
        else:
            if isinstance(y_val, complex):
                if y_val.imag != 0:
                    warnings.warn(
                        "Complex y_val passed but options.find_complex is False. "
                        "The imaginary part of y_val will be ignored.",
                        UserWarning
                    )
                target_y = float(y_val.real)
            else:
                target_y = float(y_val)

            if use_cuda and _CUPY_AVAILABLE:
                return self._solve_x_cuda(target_y, options)
            else:
                if use_cuda:
                    warnings.warn(
                        "use_cuda=True was specified, but CuPy is not installed. "
                        "Falling back to NumPy (CPU). For GPU acceleration, "
                        "install with 'pip install polysolve[cuda]'.",
                        UserWarning
                    )
        
                return self._solve_x_numpy(target_y, options)

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

        # Pre-calculate indices for slicing the destination array
        idx_elite_end = elite_size
        idx_cross_end = idx_elite_end + crossover_size
        idx_mut_end = idx_cross_end + mutation_size

        bound = _get_cauchy_bound(self.coefficients)
        min_r = -bound
        max_r = bound

        # Create initial random solutions
        src_solutions = np.random.uniform(min_r, max_r, data_size)
        dst_solutions = np.empty(data_size, dtype=np.float64)

        # Pre-allocate ranks array
        ranks = np.empty(data_size, dtype=np.float64)

        for _ in range(options.num_of_generations):
            # Calculate fitness for all solutions (vectorized)
            _calculate_ranks_numba(src_solutions, self.coefficients, y_val, ranks)

            parent_pool_size = int(data_size * options.selection_percentile)

            # 1. Get indices for the elite solutions (O(N) operation)
            #    We find the 'elite_size'-th largest element.
            elite_indices = np.argpartition(-ranks, elite_size)[:elite_size]
            
            # 2. Get indices for the parent pool (O(N) operation)
            #    We find the 'parent_pool_size'-th largest element.
            parent_pool_indices = np.argpartition(-ranks, parent_pool_size)[:parent_pool_size]

            # --- Create the next generation ---

            # 1. Elitism: Keep the best solutions as-is
            dst_solutions[:elite_size] = src_solutions[elite_indices]

            # 2. Crossover: Breed two parents to create a child
            # Select from the fitter PARENT POOL
            parents1 = src_solutions[np.random.choice(parent_pool_indices, crossover_size)]
            parents2 = src_solutions[np.random.choice(parent_pool_indices, crossover_size)]
            
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
            dst_solutions[idx_elite_end:idx_cross_end] = np.random.uniform(new_min, new_max)

            # 3. Mutation:
            # Select from the full list (indices 0 to data_size-1)
            mutation_candidates = src_solutions[np.random.randint(0, data_size, mutation_size)]
            
            # Use mutation_strength
            noise = np.random.normal(0, options.mutation_strength, mutation_size)
            dst_solutions[idx_cross_end:idx_mut_end] = mutation_candidates * (1.0 + noise)

            # 4. New Randoms: Add new blood to prevent getting stuck
            dst_solutions[idx_mut_end:] = np.random.uniform(min_r, max_r, random_size)
            
            # Assemble the new generation
            src_solutions, dst_solutions = dst_solutions, src_solutions

        # --- Final Step: Return the best results ---
        # After all generations, do one last ranking to find the best solutions
        _calculate_ranks_numba(src_solutions, self.coefficients, y_val, ranks)
        
        # 1. Define quality based on the user's desired precision
        #    (e.g., precision=5 -> rank > 1e6, precision=8 -> rank > 1e9)
        #    We add +1 for a buffer, ensuring we only get high-quality roots.
        quality_threshold = 10**(options.root_precision + 1)

        # 2. Get all solutions that meet this quality threshold
        high_quality_solutions = src_solutions[ranks > quality_threshold]

        if high_quality_solutions.size == 0:
            # No roots found that meet the quality, return empty
            return np.array([])
        
        # 3. Cluster these high-quality solutions by rounding
        rounded_solutions = np.round(high_quality_solutions, options.root_precision)

        # 4. Return only the unique roots
        unique_roots = np.unique(rounded_solutions)
        
        return np.sort(unique_roots)
    

    def _solve_complex_numpy(self, y_val: complex, options: GA_Options) -> np.ndarray:
        elite_ratio = options.elite_ratio
        crossover_ratio = options.crossover_ratio
        mutation_ratio = options.mutation_ratio
        
        data_size = options.data_size
        
        elite_size = int(data_size * elite_ratio)
        crossover_size = int(data_size * crossover_ratio)
        mutation_size = int(data_size * mutation_ratio)
        random_size = data_size - elite_size - crossover_size - mutation_size

        # Pre-calculate indices for slicing the destination array
        idx_elite_end = elite_size
        idx_cross_end = idx_elite_end + crossover_size
        idx_mut_end = idx_cross_end + mutation_size

        bound = _get_cauchy_bound(self.coefficients)
        min_r = -bound
        max_r = bound

        # 3. Initialize Population (Complex128)
        real_part = np.random.uniform(min_r, max_r, data_size)
        imag_part = np.random.uniform(min_r, max_r, data_size)
        src_solutions = real_part + 1j * imag_part

        dst_solutions = np.empty(data_size, dtype=np.complex128)

        # Cast coefficients to complex128 for Numba compatibility
        coeffs_complex = self.coefficients.astype(np.complex128)
        ranks = np.empty(data_size, dtype=np.float64)

        for _ in range(options.num_of_generations):
            # Calculate fitness for all solutions (vectorized)
            _calculate_ranks_complex_numba(src_solutions, coeffs_complex, y_val, ranks)

            parent_pool_size = int(data_size * options.selection_percentile)

            # 1. Get indices for the elite solutions (O(N) operation)
            #    We find the 'elite_size'-th largest element.
            elite_indices = np.argpartition(-ranks, elite_size)[:elite_size]
            
            # 2. Get indices for the parent pool (O(N) operation)
            #    We find the 'parent_pool_size'-th largest element.
            parent_pool_indices = np.argpartition(-ranks, parent_pool_size)[:parent_pool_size]

            # --- Create the next generation ---

            # 1. Elitism: Keep the best solutions as-is
            dst_solutions[:elite_size] = src_solutions[elite_indices]

            # 2. Crossover: Breed two parents to create a child
            # Select from the fitter PARENT POOL
            p1 = src_solutions[np.random.choice(parent_pool_indices, crossover_size)]
            p2 = src_solutions[np.random.choice(parent_pool_indices, crossover_size)]
            
            # Calculate difference vectors
            diff_real = p2.real - p1.real
            diff_imag = p2.imag - p1.imag

            alpha = options.blend_alpha

            # Generate independant weights for Real and Imaginary parts
            # This creates a 2D search area instead of a 1D
            u_real = np.random.uniform(-alpha, 1.0 + alpha, crossover_size)
            u_imag = np.random.uniform(-alpha, 1.0 + alpha, crossover_size)

            child_real = p1.real + (u_real * diff_real)
            child_imag = p1.imag + (u_imag * diff_imag)

            dst_solutions[idx_elite_end:idx_cross_end] = child_real + 1j * child_imag

            # 3. Mutation:
            # Select from the full list (indices 0 to data_size-1)
            mut_candidates = src_solutions[np.random.randint(0, data_size, mutation_size)]

            noise_real = np.random.normal(0, options.mutation_strength, mutation_size)
            noise_imag = np.random.normal(0, options.mutation_strength, mutation_size)
            
            dst_solutions[idx_cross_end:idx_mut_end] = (mut_candidates.real * (1.0 + noise_real)) + 1j * (mut_candidates.imag * (1.0 + noise_imag))

            # 4. New Randoms: Add new blood to prevent getting stuck
            rand_real = np.random.uniform(min_r, max_r, random_size)
            rand_imag = np.random.uniform(min_r, max_r, random_size)
            dst_solutions[idx_mut_end:] = rand_real + 1j * rand_imag
            
            # Assemble the new generation
            src_solutions, dst_solutions = dst_solutions, src_solutions

        # 5. Final Ranking & Clustering
        _calculate_ranks_complex_numba(src_solutions, coeffs_complex, y_val, ranks)
        quality_threshold = 10**(options.root_precision + 1)
        high_quality_solutions = src_solutions[ranks > quality_threshold]

        if high_quality_solutions.size == 0: return np.array([])

        # Rounding complex numbers: round real and imag separately
        rounded_real = np.round(high_quality_solutions.real, options.root_precision)
        rounded_imag = np.round(high_quality_solutions.imag, options.root_precision)
        
        return np.unique(rounded_real + 1j * rounded_imag)


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
        
        bound = _get_cauchy_bound(self.coefficients)
        min_r = -bound
        max_r = bound

        # Create initial random solutions on the GPU
        d_src_solutions = cupy.random.uniform(
            min_r, max_r, data_size, dtype=cupy.float64
        )

        d_dst_solutions = cupy.empty(data_size, dtype=cupy.float64)

        d_ranks = cupy.empty(data_size, dtype=cupy.float64)

        d_coefficients = cupy.array(self.coefficients, dtype=cupy.float64)

        # Calculate Shared Memory Size
        num_coeffs = len(self.coefficients)
        required_shared_mem_bytes = num_coeffs * 8

        device = cupy.cuda.Device()
        max_shared_mem = device.attributes['MaxSharedMemoryPerBlock']

        use_shared_mem = True

        if required_shared_mem_bytes > max_shared_mem:
            # The polynomial is too big for the cache!
            # We must fall back to the slower Global Memory kernel to prevent a crash.
            use_shared_mem = False
            warnings.warn(
                f"Polynomial degree ({num_coeffs}) exceeds GPU Shared Memory limit "
                f"({max_shared_mem} bytes). Falling back to Global Memory (slower).",
                UserWarning
            )

        # Kernel Setup
        if use_shared_mem:
            fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL_FLOAT_DYNAMIC, 'fitness_kernel_shared')
            kwargs = {'shared_mem': required_shared_mem_bytes}
        else:
            fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL_FLOAT, 'fitness_kernel')
            kwargs = {}
            
        threads_per_block = 512
        blocks_per_grid = (options.data_size + threads_per_block - 1) // threads_per_block

        # Indices for slicing the destination buffer
        idx_elite_end = elite_size
        idx_cross_end = idx_elite_end + crossover_size
        idx_mut_end = idx_cross_end + mutation_size

        for i in range(options.num_of_generations):
            # Run the fitness kernel on the GPU
            
            fitness_gpu(
                (blocks_per_grid,), (threads_per_block,),
                (d_coefficients, d_coefficients.size, d_src_solutions, d_ranks, d_src_solutions.size, y_val),
                **kwargs
            )
            
            # Sort solutions by rank on the GPU
            sorted_indices = cupy.argsort(-d_ranks)
            d_sorted_src_solutions = d_src_solutions[sorted_indices]
            
            # --- Create the next generation ---
            
            # 1. Elitism
            d_dst_solutions[:elite_size] = d_sorted_src_solutions[:elite_size]

            # 2. Crossover
            parent_pool_size = int(data_size * options.selection_percentile)
            # Select from the fitter PARENT POOL
            p1_indices = cupy.random.randint(0, parent_pool_size, crossover_size)
            p2_indices = cupy.random.randint(0, parent_pool_size, crossover_size)
            # Get parents directly from the sorted solutions array using the pool-sized indices
            d_p1 = d_sorted_src_solutions[p1_indices]
            d_p2 = d_sorted_src_solutions[p2_indices]
            
            # Blend Crossover (BLX-alpha)
            alpha = options.blend_alpha

            diff = d_p2 - d_p1
            u = cupy.random.uniform(-alpha, 1.0 + alpha, crossover_size)

            d_dst_solutions[idx_elite_end:idx_cross_end] = d_p1 + (u * diff)

            # 3. Mutation
            # Select from the full list (indices 0 to data_size-1)
            mutation_indices = cupy.random.randint(0, data_size, mutation_size)
            d_mutation_candidates = d_sorted_src_solutions[mutation_indices]
            
            # Use mutation_strength
            noise = cupy.random.normal(0, options.mutation_strength, mutation_size)
            d_dst_solutions[idx_cross_end:idx_mut_end] = d_mutation_candidates * (1.0 + noise)

            # 4. New Randoms
            d_dst_solutions[idx_mut_end:] = cupy.random.uniform(
                min_r, max_r, random_size, dtype=cupy.float64
            )

            # Assemble the new generation
            # d_dst becomes the new source for the next generation
            d_src_solutions, d_dst_solutions = d_dst_solutions, d_src_solutions

        # --- Final Step: Return the best results ---
        # After all generations, do one last ranking to find the best solutions
        fitness_gpu(
            (blocks_per_grid,), (threads_per_block,),
            (d_coefficients, d_coefficients.size, d_src_solutions, d_ranks, d_src_solutions.size, y_val),
            **kwargs
        )
        
        # 1. Define quality based on the user's desired precision
        #    (e.g., precision=5 -> rank > 1e6, precision=8 -> rank > 1e9)
        #    We add +1 for a buffer, ensuring we only get high-quality roots.
        quality_threshold = 10**(options.root_precision + 1)
        
        # 2. Get all solutions that meet this quality threshold
        d_high_quality_solutions = d_src_solutions[d_ranks > quality_threshold]

        if d_high_quality_solutions.size == 0:
            return np.array([])
            
        # 3. Cluster these high-quality solutions on the GPU by rounding
        d_rounded_solutions = cupy.round(d_high_quality_solutions, options.root_precision)
        
        # 4. Get only the unique roots
        d_unique_roots = cupy.unique(d_rounded_solutions)

        # Sort the unique roots and copy back to CPU
        final_solutions_gpu = cupy.sort(d_unique_roots)
        return final_solutions_gpu.get()


    def _solve_complex_cuda(self, y_val: complex, options: GA_Options) -> np.ndarray:
        elite_ratio = options.elite_ratio
        crossover_ratio = options.crossover_ratio
        mutation_ratio = options.mutation_ratio
        
        data_size = options.data_size
        
        elite_size = int(data_size * elite_ratio)
        crossover_size = int(data_size * crossover_ratio)
        mutation_size = int(data_size * mutation_ratio)
        random_size = data_size - elite_size - crossover_size - mutation_size

        # 1. Prepare Coefficients (Split into Real/Imag for the Kernel)
        # We pass real and imag arrays separately to avoid struct alignment issues
        coeffs = self.coefficients.astype(np.complex128)
        d_coeffs_real = cupy.array(coeffs.real, dtype=cupy.float64)
        d_coeffs_imag = cupy.array(coeffs.imag, dtype=cupy.float64)

        d_y_real = cupy.float64(y_val.real)
        d_y_imag = cupy.float64(y_val.imag)

        bound = _get_cauchy_bound(self.coefficients)
        min_r = -bound
        max_r = bound

        real_part = cupy.random.uniform(min_r, max_r, data_size, dtype=cupy.float64)
        imag_part = cupy.random.uniform(min_r, max_r, data_size, dtype=cupy.float64)
        d_src_solutions = real_part + 1j * imag_part

        d_dst_solutions = cupy.empty(data_size, dtype=cupy.complex128)
        d_ranks = cupy.empty(data_size, dtype=cupy.float64)

        # Calculate Shared Memory Size
        num_coeffs = len(self.coefficients)
        required_shared_mem_bytes = (num_coeffs * 8) * 2

        device = cupy.cuda.Device()
        max_shared_mem = device.attributes['MaxSharedMemoryPerBlock']

        use_shared_mem = True

        if required_shared_mem_bytes > max_shared_mem:
            # The polynomial is too big for the cache!
            # We must fall back to the slower Global Memory kernel to prevent a crash.
            use_shared_mem = False
            warnings.warn(
                f"Polynomial degree ({num_coeffs}) exceeds GPU Shared Memory limit "
                f"({max_shared_mem} bytes). Falling back to Global Memory (slower).",
                UserWarning
            )

        # Kernel Setup
        if use_shared_mem:
            fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL_COMPLEX_DYNAMIC, 'fitness_kernel_complex_shared')
            kwargs = {'shared_mem': required_shared_mem_bytes}
        else:
            fitness_gpu = cupy.RawKernel(_FITNESS_KERNEL_COMPLEX, 'fitness_kernel_complex')
            kwargs = {}
            
        threads_per_block = 512
        blocks_per_grid = (options.data_size + threads_per_block - 1) // threads_per_block

        idx_elite_end = elite_size
        idx_cross_end = idx_elite_end + crossover_size
        idx_mut_end = idx_cross_end + mutation_size

        for _ in range(options.num_of_generations):
            d_real_cont = cupy.ascontiguousarray(d_src_solutions.real)
            d_imag_cont = cupy.ascontiguousarray(d_src_solutions.imag)

            fitness_gpu(
                (blocks_per_grid,), (threads_per_block,),
                (d_coeffs_real, d_coeffs_imag, d_coeffs_real.size, 
                 d_real_cont, d_imag_cont, d_ranks, data_size,
                 d_y_real, d_y_imag),
                **kwargs
            )

            # Sort (using d_ranks)
            sorted_indices = cupy.argsort(-d_ranks)
            d_sorted_src_solutions = d_src_solutions[sorted_indices]

            # 1. Elite: Keep the best
            d_dst_solutions[:elite_size] = d_sorted_src_solutions[:elite_size]

            # 2. Crossover: Blend Crossover (BLX-alpha)
            #    Select parents from the top percentile
            parent_pool_size = int(data_size * options.selection_percentile)
            
            #    Randomly pair parents
            p1_indices = cupy.random.randint(0, parent_pool_size, crossover_size)
            p2_indices = cupy.random.randint(0, parent_pool_size, crossover_size)

            p1 = d_sorted_src_solutions[p1_indices]
            p2 = d_sorted_src_solutions[p2_indices]

            # Calculate difference vectors
            diff_real = p2.real - p1.real
            diff_imag = p2.imag - p1.imag

            alpha = options.blend_alpha

            # Generate independant weights for Real and Imaginary parts
            # This creates a 2D search area instead of a 1D
            u_real = cupy.random.uniform(-alpha, 1.0 + alpha, crossover_size)
            u_imag = cupy.random.uniform(-alpha, 1.0 + alpha, crossover_size)

            child_real = p1.real + (u_real * diff_real)
            child_imag = p1.imag + (u_imag * diff_imag)
            
            # Apply Crossover
            d_dst_solutions[idx_elite_end:idx_cross_end] = child_real + 1j * child_imag

            # 3. Mutation: Perturb existing solutions
            #    Pick random candidates from the full population
            mut_indices = cupy.random.randint(0, data_size, mutation_size)
            mut_candidates = d_sorted_src_solutions[mut_indices]

            #    Generate Independent Scaling Factors for Real and Imaginary parts
            #    Range: [1 - strength, 1 + strength]
            noise_real = cupy.random.normal(0, options.mutation_strength, mutation_size)
            noise_imag = cupy.random.normal(0, options.mutation_strength, mutation_size)

            #    Apply Mutation: Scale Real/Imag independently to allow "rotation" off the line
            d_dst_solutions[idx_cross_end:idx_mut_end] = (mut_candidates.real * (1.0 + noise_real)) + 1j * (mut_candidates.imag * (1.0 + noise_imag))

            # 4. Random Injection: Fresh genetic material
            rand_real = cupy.random.uniform(min_r, max_r, random_size, dtype=cupy.float64)
            rand_imag = cupy.random.uniform(min_r, max_r, random_size, dtype=cupy.float64)
            d_dst_solutions[idx_mut_end:] = rand_real + 1j * rand_imag

            d_src_solutions, d_dst_solutions = d_dst_solutions, d_src_solutions

        d_real_cont = cupy.ascontiguousarray(d_src_solutions.real)
        d_imag_cont = cupy.ascontiguousarray(d_src_solutions.imag)

        # Final Rank
        fitness_gpu(
            (blocks_per_grid,), (threads_per_block,),
            (d_coeffs_real, d_coeffs_imag, d_coeffs_real.size,
             d_real_cont, d_imag_cont, d_ranks, data_size,
             d_y_real, d_y_imag),
            **kwargs
        )

        # Filtering & Return
        quality_threshold = 10**(options.root_precision + 1)
        d_high_quality_solutions = d_src_solutions[d_ranks > quality_threshold]
        
        if d_high_quality_solutions.size == 0: return np.array([])

        rounded_real = cupy.round(d_high_quality_solutions.real, options.root_precision)
        rounded_imag = cupy.round(d_high_quality_solutions.imag, options.root_precision)

        d_unique = cupy.unique(rounded_real + 1j * rounded_imag)
        
        # Sort the unique roots and copy back to CPU
        final_solutions_gpu = cupy.sort(d_unique)
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
        new_coefficients = self._strip_leading_zeros(new_coefficients)
        
        result_func = Function(len(new_coefficients) - 1)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func
    
    def _strip_leading_zeros(self, coeffs: np.ndarray) -> np.ndarray:
        # Remove leading zeros
        while len(coeffs) > 1 and np.isclose(coeffs[0], 0):
            coeffs = coeffs[1:]
        return coeffs

    def __sub__(self, other: 'Function') -> 'Function':
        """Subtracts another Function object from this one."""
        self._check_initialized()
        other._check_initialized()

        new_coefficients = np.polysub(self.coefficients, other.coefficients)
        new_coefficients = self._strip_leading_zeros(new_coefficients)
        
        result_func = Function(len(new_coefficients) - 1)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func
    
    def _multiply_by_scalar(self, scalar: Union[int, float, complex]) -> 'Function':
        """Helper method to multiply the function by a scalar constant."""
        self._check_initialized()

        if scalar == 0:
            result_func = Function(0)
            result_func.set_coeffs([0])
            return result_func
    
        new_coefficients = self.coefficients * scalar
        new_coefficients = self._strip_leading_zeros(new_coefficients)
    
        result_func = Function(self._largest_exponent)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func

    def _multiply_by_function(self, other: 'Function') -> 'Function':
        """Helper method for polynomial multiplication (Function * Function)."""
        self._check_initialized()
        other._check_initialized()

        # np.polymul performs convolution of coefficients to multiply polynomials
        new_coefficients = np.polymul(self.coefficients, other.coefficients)
        new_coefficients = self._strip_leading_zeros(new_coefficients)
    
        # The degree of the resulting polynomial is derived from the new coefficients
        new_degree = len(new_coefficients) - 1
    
        result_func = Function(new_degree)
        result_func.set_coeffs(new_coefficients.tolist())
        return result_func
        
    def __mul__(self, other: Union['Function', int, float, complex]) -> 'Function':
        """Multiplies the function by a scalar constant."""
        if isinstance(other, (int, float)):
            return self._multiply_by_scalar(other)
        elif isinstance(other, self.__class__):
            return self._multiply_by_function(other)
        else:
            return NotImplemented

    def __rmul__(self, scalar: Union[int, float, complex]) -> 'Function':
        """Handles scalar multiplication from the right (e.g., 3 * func)."""

        return self.__mul__(scalar)
        
    def __imul__(self, other: Union['Function', int, float, complex]) -> 'Function':
        """Performs in-place multiplication by a scalar (func *= 3)."""

        self._check_initialized()
    
        if isinstance(other, (int, float, complex)):
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
    
    def __eq__(self, other: object) -> bool:
        """
        Checks if two Function objects are equal by comparing
        their coefficients.
        """
        # Check if the 'other' object is even a Function
        if not isinstance(other, Function):
            return NotImplemented
        
        # Ensure both are initialized before trying to access .coefficients
        if not self._initialized or not other._initialized:
            return False

        c1 = self._strip_leading_zeros(self.coefficients)
        c2 = self._strip_leading_zeros(other.coefficients)
        
        if c1.shape != c2.shape:
            return False
            
        return np.allclose(c1, c2)


    def quadratic_solve(self) -> Optional[List[Union[complex, float]]]:
        """
        Calculates the roots (real or complex) of a quadratic function.

        Returns:
            Optional[List[complex]]: A list containing the two roots
        """
        self._check_initialized()
        if self.largest_exponent != 2:
            raise ValueError("Input function must be quadratic (degree 2) to use quadratic_solve.")

        a, b, c = self.coefficients

        discriminant = (b**2) - (4*a*c)

        sqrt_discriminant = cmath.sqrt(discriminant)

        if b >= 0:
            sign_b = 1
        else:
            sign_b = -1
        
        root1 = (-b - sign_b * sqrt_discriminant) / (2 * a)

        if abs(root1) < 1e-15:
            root2 = (-b + sign_b * sqrt_discriminant) / (2 * a)
        else:
            # Standard case: Use Vieta's formula
            root2 = (c / a) / root1
        
        roots = np.array([root1, root2])
        roots.sort()

        if np.all(np.abs(roots.imag) < 1e-15):
            return roots.real.astype(np.float64)
    
        return roots

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

    fc = Function(2, coefficients=[1, 2, 2])
    print(f"\nFunction fc: {f1}")

    # --- Root Finding ---
    # 1. Analytical solution for quadratic
    roots_analytic = f1.quadratic_solve()
    print(f"\nAnalytic roots of f1: {roots_analytic}") # Expected: -1, 2.5
    c_roots_analytic = fc.quadratic_solve()
    print(f"Analytic roots of fc: {c_roots_analytic}") # Expected: -1-j, -1+j

    # 2. Genetic algorithm solution
    ga_opts = GA_Options(num_of_generations=100, data_size=100000, root_precision=3, selection_percentile=0.75)
    print("\nFinding real roots with Genetic Algorithm (CPU)...")
    roots_ga_cpu = f1.get_real_roots(ga_opts)
    print(f"Approximate real roots from GA (CPU): {roots_ga_cpu}")
    print("\nFinding all roots of fc with Genetic Algorithm (CPU)...")
    c_roots_ga_cpu = fc.get_roots(ga_opts)
    print(f"Approximate roots of fc from GA (CPU): {c_roots_ga_cpu}")

    # 3. CUDA accelerated genetic algorithm
    if _CUPY_AVAILABLE:
        print("\nFinding real roots with Genetic Algorithm (GPU)...")
        # Since this PC has an RTX 4060 Ti, we can use the CUDA version.
        roots_ga_gpu = f1.get_real_roots(ga_opts, use_cuda=True)
        print(f"Approximate real roots from GA (GPU): {roots_ga_gpu}")
        print("\nFinding all roots of fc with Genetic Algorithm (GPU)...")
        c_roots_ga_gpu = fc.get_roots(ga_opts)
        print(f"Approximate roots of fc from GA (GPU): {c_roots_ga_gpu}")
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
