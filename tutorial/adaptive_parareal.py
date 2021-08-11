from typing import List, Optional, Tuple
import numpy as np
from multiprocessing import Pool, pool
from abc import ABC, abstractmethod

import functools
import os.path
import shutil
import pickle
import hashlib

class BaseParareal(ABC):
    def __init__(self,
                 a: float,
                 b: float,
                 n_coarse: int,
                 iterations: int,
                 x_initial: np.ndarray,
                 ):
        self.var_shape = x_initial.shape
        self.x0 = x_initial
        self.n_coarse = n_coarse
        self.iterations = iterations
        
        self.print_name: Optional[str] = None
        self.tol: Optional[float] = None
        self.save_fine = True
        self.solved = False
        self.iters_taken: Optional[int] = None
        
        self.t_coarse = self.get_coarse_t(a, b)
        self.t_fine = np.empty((self.n_coarse+1, self.iterations), object)
        self.t_fine.fill([])
        
        self._x_coarse = np.empty((self.n_coarse+1, iterations, *self.var_shape))
        x0_repeated = self.x0.reshape((1, *self.var_shape)).repeat(iterations, 0)
        self._x_coarse[0, :] = x0_repeated
        self.x_coarse_corr = self._x_coarse.copy()
        
        self.x_fine = np.empty((self.n_coarse, self.iterations), object)
        self.x_fine.fill([])
        
    def solve(self, tolerance: Optional[float] = None, processors: Optional[int] = None,
              print_ref: Optional[str] = None, save_fine: bool = True, always_solve = False):
        """Solve the parareal problem
        
        Parameters:
        tolerance: float or None
            If this tolerance is reached and is_within_tol is given the solve
            wil terminate. If None the solve will complete all iterations
        processors: int or None
            Number of parallel processes to run. If None the solve will be done
            in serial
        print_ref: str or None
            If None progress information will not be displayed. Otherwise the
            string will be appended to the start of progress messages (can be 
            an empty string)
        save_fine: bool = True
            Whether to save the value of the function at each fine point
        always_solve: bool = False
            Whether to solve even if the system is already solved (as indicated
            by self.solved)
            
        Returns:
        iterations_done: int
            Number of iterations completed. This will normally be self.iterations-1
            but can be less if the tolerance is reached.
            Note: Any values in arrays with iteration index higher than iterations_done
            will be uninitialised and should not be used.
        """
        # Don't solve again if already done with the same tolerance and with fine
        # values saved if required
        if ( not always_solve and self.solved and tolerance == self.tol and
             not (save_fine and not self.save_fine) ):
            self._print('Solve already completed')
            return self.iters_taken
        
        self.print_name = print_ref
        self.tol = tolerance
        self.save_fine = save_fine
        self._print('Starting parareal solve')
        if processors is None:
            return self._solve()
        try:
            # Create a pool of subprocesses, each with access to the class
            self._print('Creating pool')
            p = Pool(processors, self._child_process_init, (self,))
            self._print('Pool created')
            return self._solve(p)
        finally:
            p.terminate()
            p.join()

    @staticmethod
    def _child_process_init(_active_obj: 'BaseParareal'):
        """Allows self to be passed to the child process on initialisation instead of every function call.
        Note: Any changes to self will not be mirrored however and so must be passed separately.
        
        See: https://stackoverflow.com/a/25830011/7511654
        """
        global active_obj
        active_obj = _active_obj # type: ignore
        
    def _print(self, *args, **kwargs):
        if self.print_name is not None:
            if self.print_name != '':
                print(f'{self.print_name}-PR: ', end='')
            print(*args, **kwargs)
            
    def _solve(self, p: Optional[pool.Pool] = None) -> int:
        self._print('Starting initial coarse solve')
        for j_coarse in range(self.n_coarse):
            next_coarse = self.coarse_integration_func(self.t_coarse[j_coarse],
                                                                   self.t_coarse[j_coarse+1],
                                                                   self._x_coarse[j_coarse, 0],
                                                                   j_coarse, 0)
            self._x_coarse[j_coarse+1, 0] = next_coarse
            self.x_coarse_corr[j_coarse+1, 0] = next_coarse
        
        for k in range(1, self.iterations):
            self._print(f'----------------- Iteration {k: <2} -----------------')
            # Solve in serial
            if p is None:
                for j in range(self.n_coarse):
                    t_result, x_result = self._do_fine_integ(j, k)
                    self.t_fine[j, k] = t_result
                    self.x_fine[j, k] = x_result
                    
                self._print('Starting coarse prediction and correction')
                for t in range(self.n_coarse):
                    self._coarse_correction(t, k)
            # Solve in parallel
            else:
                results = p.starmap(BaseParareal._parallel_integ, ((j, k, self.x_coarse_corr[j, k-1]) for j in range(self.n_coarse)))
                self._print('Starting coarse prediction and correction')
                for t in range(self.n_coarse):
                    t_result, x_result = results[t]
                    self.t_fine[t, k] = t_result
                    self.x_fine[t, k] = x_result
                    self._coarse_correction(t, k)
                    
            # Check if the desired tolerance has been achieved
            if self.tol is not None and self.is_within_tol(self.x_coarse_corr[:, k], self.x_coarse_corr[:, k-1]):
                self._print('Achieved given tolerance after iteration', k)
                self.solved = True
                self.iters_taken = k
                return k
        self.iters_taken = k
        self.solved = True
        return k
                        
    @staticmethod
    def _parallel_integ(j_coarse, k_iteration, x_initial):
        return active_obj._do_fine_integ(j_coarse, k_iteration, x_initial)
    
    def _do_fine_integ(self, j_coarse, k_iteration, x_initial=None):
        # Allow x_initial to be specified differently to the class
        # This allows it to be passed to the subprocess separately
        if x_initial is None:
            x_initial = self.x_coarse_corr[j_coarse, k_iteration-1]
        self._print(f'k={k_iteration: >2} Starting integration for coarse step {j_coarse}\n', end='')
        
        t_fine_result, integ_result = self.fine_integration(
            self.t_coarse[j_coarse],self.t_coarse[j_coarse+1], x_initial, j_coarse, k_iteration)
        
        self._print(f'k={k_iteration: >2} Done integration for coarse step {j_coarse}\n', end='')
        return (t_fine_result, integ_result)
        
    def _coarse_correction(self, t_coarse, k_iteration):
        self._x_coarse[t_coarse+1, k_iteration] = self.coarse_integration_func(
            self.t_coarse[t_coarse], self.t_coarse[t_coarse+1], self.x_coarse_corr[t_coarse, k_iteration],
            t_coarse, k_iteration)
        self.x_coarse_corr[t_coarse+1, k_iteration] = self._x_coarse[t_coarse+1, k_iteration] -\
            self._x_coarse[t_coarse+1, k_iteration-1] + self.x_fine[t_coarse, k_iteration][-1]
        
    # Methods that can/must be overridden
    def get_coarse_t(self, a: float, b: float) -> np.ndarray:
        """Returns a numpy array of the coarse t values.
        Can be overridden for non-uniform coarse steps
        
        Returns:
        t_coarse: np.ndarray
            Array of t values for the coarse integrator to be calculated at. Must be of length
            self.n_coarse+1 and start/end with a/b respectively
        """
        return np.linspace(a, b, self.n_coarse+1)
    
    def is_within_tol(self, x_current: np.ndarray, x_previous: np.ndarray) -> bool:
        """Checks if the parareal solve has reached the desired tolerance and can stop.
        It can be assumed self.tol has a value
        By default always returns False but can be overridden
        
        Returns:
        within_tol: bool
            Has the solve reached tolerance and so should stop
        """
        return False
    
    @abstractmethod
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        """Coarsely integrates x_in by one step.
        Must be overridden in a subclass.
        
        Returns:
        x_coarse: np.ndarray
            The value of the function at t=b
        """
        raise NotImplementedError
    
    @abstractmethod
    def fine_integration(self, t_start: float, t_end: float, x_initial: np.ndarray,
                   coarse_step: int, iteration: int) -> Tuple[Optional[List[float]], List[np.ndarray]]:
        """Finely integrates x_initial in as many steps as is desired.
        Must be overridden in a subclass.
        
        Returns:
        t_fine_result: List[float]
            List containing the values of each fine step used. Should be None if
            self.save_fine is False
        x_fine_result: List[np.ndarray]
            If self.save_fine is True then contains a list of the values of the
            function at each time step. Must be of the same length as t_fine_result.
            If self.save_fine is False then returns a single item list with the
            value of the function at t_end
        """
        raise NotImplementedError

class FixedParareal(BaseParareal):
    def __init__(self, a: float, b: float, n_coarse: int, n_fine: int, iterations: int, x_initial: np.ndarray):
        super().__init__(a, b, n_coarse, iterations, x_initial)
        self.n_fine = n_fine
    
    def fine_integration(self, t_start: float, t_end: float, x_initial: np.ndarray,
                         coarse_step: int, iteration: int) -> Tuple[Optional[List[float]], List[np.ndarray]]:
        n_fine = self.n_fine + 1
        # Generate list of t for this step
        t_fine_result = list(np.linspace(t_start, t_end, n_fine))
        # Do the integration
        integ_result = self.fine_integration_func(t_fine_result, x_initial)
        
        if self.save_fine:
            return (t_fine_result, integ_result)
        else:
            return (None, integ_result)
    
    @abstractmethod
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        """If self.save_fine is True, returns the value of the function at each of t_vals.
        The returned list should be of the same length as t_vals and contain arrays with the
        same shape as x_in.
        If self.save_fine is False returns the value of the function at t_vals[-1]
        """
        raise NotImplementedError
    
    
class CachedPR:
    """Mixin for Parareal objects to provide caching of results between runs"""
    def get_cache(self, *args, quiet_output=False):
        """Finds a cached solve if it exists and updates the object with the cached values.
        
        This should be called at the start of __init__ and provided all arguments relevant
        to the solve. A solve with the same argument hash is then searched for with the
        results applied to the object. If a cached solution is found - indicated by found_cache,
        initiation should be aborted to avoid overwriting the solve.
        
        Keyword arguments:
        quiet_output: bool = False
            Whether to suppress printed output
        
        Returns:
        found_cache: bool
            Was a cached result found and loaded successfully. If True, no further initiation
            should take place.
        """
        self._quiet = quiet_output
        
        cls_name = self.__class__.__name__
        folder = f'tutorial\\pickles\\{cls_name}'
        if not self._quiet:
            print('Checking for cached solve')
        if not os.path.exists(folder):
            if not self._quiet:
                print(f'Cache folder {folder} not found')
            return False
        
        self._args_hash = self._make_hash(args)
        if not self._quiet:
            print('Parameter hash:', self._args_hash)
        if os.path.exists(f'{folder}\\{self._args_hash}'):
            # Cache exists
            new_obj = self.load_cache(self._args_hash, self._quiet)
            self.__dict__.update(new_obj.__dict__)
            return True
        
        # No cache exists
        if not self._quiet:
            print('No cache found')
        return False
    
    @staticmethod
    def _make_hash(obj):
        if callable(obj):
            if isinstance(obj, functools.partial):
                return CachedPR._make_hash((*obj.args, obj.keywords, obj.func.__name__))
            return CachedPR._hash_func(obj.__name__)
        if isinstance(obj, (set, tuple, list, np.ndarray)):
            return CachedPR._hash_func(tuple(CachedPR._make_hash(item) for item in obj))
        if isinstance(obj, dict):
            new_dict = {}
            for name, v in obj.items():
                new_dict[name] = CachedPR._make_hash(v)
            return CachedPR._hash_func(sorted(new_dict.items()))
        return CachedPR._hash_func(obj)
    
    @staticmethod
    def _hash_func(obj):
        byte_obj = bytes(str(obj), 'utf-8')
        return hashlib.md5(byte_obj).hexdigest()
    
    def save_cache(self, name: Optional[str] = None):
        """Cache the current state of the object. This is done using the argument
        hash as the file name. If name is provided a second copy is stored with as
        well.
        
        Parameters:
        name: str or None
            The name to save the object under (as well as the hash value)
        """
        cls_name = self.__class__.__name__
        folder = f'tutorial\\pickles\\{cls_name}'
        if not self._quiet:
            print(f'Saving cache to {folder}\\{self._args_hash}')
        if not os.path.exists(folder):
            raise FileNotFoundError('Caching folder does not exist')

        with open(f'{folder}\\{self._args_hash}', 'wb') as f:
            pickle.dump(self, f)
        if name:
            shutil.copyfile(f'{folder}\\{self._args_hash}', f'{folder}\\{name}')
            
    @classmethod
    def load_cache(cls, file_name: str, quiet=False):
        """Load and return a saved parareal solve."""
        if not quiet:
            print(f'Loading cache at tutorial\\pickles\\{cls.__name__}\\{file_name}')
        with open(f'tutorial\\pickles\\{cls.__name__}\\{file_name}', 'rb') as f:
            return pickle.load(f)
    

# Example implementation for the Lorenz system
class _PRLorenz(FixedParareal):
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        return RK4(lorenz63, b-a, 2, x_in)[-1] # type: ignore
    
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        dt = t_vals[1] - t_vals[0]
        result = RK4(lorenz63, dt, len(t_vals), x_in) # type: ignore
        if self.save_fine:
            return list(result)
        else:
            return [result[-1]]
    
if __name__ == '__main__':
    from brusselator import RK4
    from lorenz63 import lorenz63
    
    lorenz = _PRLorenz(0, 5, 40, 36, 10, np.array([5,-5,20]))
    lorenz.solve()
    
    from pr_animation import PRanimation3D
    import matplotlib.cm as cm
    
    animator = PRanimation3D(lorenz.x_coarse_corr, lorenz.x_fine, [[-20,20], [-25,25], [0,40]],
                             ['x', 'y', 'z'], 8, 1, title='Lorenz attractor',
                             line_colour=cm.get_cmap('YlOrRd_r'), dot_colour=cm.get_cmap('YlOrRd_r'))
    animator.animate('test.gif', 15)