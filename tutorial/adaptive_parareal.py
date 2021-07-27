from typing import List, Optional, Tuple
import numpy as np
from multiprocessing import Pool, pool
from abc import ABC, abstractmethod

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
        
        self.pool = None
        self.print_output = False
        
        self.t_coarse = self.get_coarse_t(a, b)
        self.t_fine = np.empty((self.n_coarse+1, self.iterations), object)
        self.t_fine.fill([])
        
        self.x_coarse = np.empty((self.n_coarse+1, iterations, *self.var_shape))
        x0_repeated = self.x0.reshape((1, *self.var_shape)).repeat(iterations, 0)
        self.x_coarse[0, :] = x0_repeated
        self.x_coarse_corr = np.empty((self.n_coarse+1, iterations, *self.var_shape))
        
        self.x_fine = np.empty((self.n_coarse, self.iterations), object)
        self.x_fine.fill([])
        
    def solve(self, processors: Optional[int] = None, print_info: bool = False):
        self.print_output = print_info
        if processors is None:
            return self._solve()
        try:
            # Create a pool of subprocesses, each with access to the class
            print('Creating pool')
            p = Pool(processors, self.child_process_init, (self,))
            print('Pool created')
            return self._solve(p)
        finally:
            p.terminate()
            p.join()

    @staticmethod
    def child_process_init(_active_obj: 'BaseParareal'):
        """Allows self to be passed to the child process on initialisation instead of every function call.
        Note: Any changes to self will not be mirrored however and so must be passed separately.
        
        See: https://stackoverflow.com/a/25830011/7511654
        """
        global active_obj
        active_obj = _active_obj # type: ignore
        
    def print(self, *args, **kwargs):
        if self.print_output:
            print(*args, **kwargs)
            
    def _solve(self, p: Optional[pool.Pool] = None):
        for j_coarse in range(self.n_coarse):
            self.x_coarse[j_coarse+1, 0] = self.coarse_integration_func(self.t_coarse[j_coarse],
                                                                   self.t_coarse[j_coarse+1],
                                                                   self.x_coarse[j_coarse, 0],
                                                                   j_coarse, 0)
            self.x_coarse_corr = self.x_coarse.copy()
        
        for k in range(1, self.iterations):
            print(f'----------------- Iteration {k: <2} -----------------')
            if p is None:
                for j in range(self.n_coarse):
                    t_result, x_result = self._do_fine_integ(j, k)
                    self.t_fine[j, k] = t_result
                    self.x_fine[j, k] = x_result
                    
                for t in range(self.n_coarse):
                    self._coarse_correction(t, k)
            else:
                results = p.starmap(BaseParareal._parallel_integ, ((j, k, self.x_coarse_corr[j, k-1]) for j in range(self.n_coarse)))
                for t in range(self.n_coarse):
                    t_result, x_result = results[t]
                    self.t_fine[t, k] = t_result
                    self.x_fine[t, k] = x_result
                    self._coarse_correction(t, k)
                        
    @staticmethod
    def _parallel_integ(j_coarse, k_iteration, x_initial):
        return active_obj._do_fine_integ(j_coarse, k_iteration, x_initial)
    
    def _do_fine_integ(self, j_coarse, k_iteration, x_initial=None):
        # Allow x_inital to be specified differently to the class
        # This allows it to be passed to the subprocess separately
        if x_initial is None:
            x_initial = self.x_coarse_corr[j_coarse, k_iteration-1]
        self.print(f'Starting integration for coarse step {j_coarse}\n', end='')
        
        t_fine_result, integ_result = self.fine_integration(
            self.t_coarse[j_coarse],self.t_coarse[j_coarse+1], x_initial, j_coarse, k_iteration)
        
        self.print(f'Done integration for coarse step {j_coarse}\n', end='')
        return (t_fine_result, integ_result)
        
    def _coarse_correction(self, t_coarse, k_iteration):
        self.x_coarse[t_coarse+1, k_iteration] = self.coarse_integration_func(
            self.t_coarse[t_coarse], self.t_coarse[t_coarse+1], self.x_coarse_corr[t_coarse, k_iteration],
            t_coarse, k_iteration)
        self.x_coarse_corr[t_coarse+1, k_iteration] = self.x_coarse[t_coarse+1, k_iteration] -\
            self.x_coarse[t_coarse+1, k_iteration-1] + self.x_fine[t_coarse, k_iteration][-1]
        
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
                   coarse_step: int, iteration: int) -> Tuple[List[float], List[np.ndarray]]:
        """Finely integrates x_initial in as many steps as is desired.
        Must be overridden in a subclass.
        
        Returns:
        t_fine_result: List[float]
            List containg the values of each fine step used
        x_fine_result: List[np.ndarray]
            List of the values of the function at each time step. Must be of the
            same length as t_fine_result.
        """
        raise NotImplementedError

class FixedParareal(BaseParareal):
    def __init__(self, a: float, b: float, n_coarse: int, n_fine: int, iterations: int, x_initial: np.ndarray):
        super().__init__(a, b, n_coarse, iterations, x_initial)
        self.n_fine = n_fine
    
    def fine_integration(self, t_start: float, t_end: float, x_initial: np.ndarray,
                         coarse_step: int, iteration: int) -> Tuple[List[float], List[np.ndarray]]:
        n_fine = self.n_fine + 1
        # Generate list of t for this step
        t_fine_result = list(np.linspace(t_start, t_end, n_fine))
        # Do the integration
        integ_result = self.fine_integration_func(t_fine_result, x_initial)
        
        return (t_fine_result, integ_result)
    
    @abstractmethod
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        """Returns the value of the function at each of t_vals.
        The returned list should be of the same length as t_vals and contain arrays with the
        same shape as x_in.
        """
        raise NotImplementedError
    
class PRLorenz(FixedParareal):
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        return RK4(lorenz63, b-a, 2, x_in)[-1] # type: ignore
    
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        dt = t_vals[1] - t_vals[0]
        return list(RK4(lorenz63, dt, len(t_vals), x_in)) # type: ignore
    
if __name__ == '__main__':
    from brusselator import RK4
    from lorenz63 import lorenz63
    
    lorenz = PRLorenz(0, 5, 40, 36, 10, np.array([5,-5,20]))
    lorenz.solve()
    
    from pr_animation import PRanimation3D
    import matplotlib.cm as cm
    
    animator = PRanimation3D(lorenz.x_coarse_corr, lorenz.x_fine, [[-20,20], [-25,25], [0,40]],
                             ['x', 'y', 'z'], 8, 1, title='Lorenz attractor',
                             line_colour=cm.get_cmap('YlOrRd_r'), dot_colour=cm.get_cmap('YlOrRd_r'))
    animator.animate('test.gif', 15)