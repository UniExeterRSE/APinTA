from typing import Callable, Optional, Sequence, List, Tuple
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import parareal as pr
from pr_animation import PRanimation2D, PRanimationAdaptive2D
from adaptive_parareal import BaseParareal, FixedParareal, CachedPR

A = 1
B = 3

def RK4(func: Callable, dt: float, n: int, x0: Sequence[float], **func_kwargs):
    param_n = len(x0)
    x_n = np.empty((n, param_n))
    x_n[0] = x0
    
    for i in range(1,n):
        k1 = np.array(func(*x_n[i-1], **func_kwargs))
        k2 = np.array(func(*(x_n[i-1] + k1*dt*0.5), **func_kwargs))
        k3 = np.array(func(*(x_n[i-1] + k2*dt*0.5), **func_kwargs))
        k4 = np.array(func(*(x_n[i-1] + k3*dt), **func_kwargs))
        x_n[i] = x_n[i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        
    return x_n

def brusselator(x, y, A=1, B=3):
    y_dot = B*x - x**2*y
    # x_dot = A + x**2*y - (B+1)*x
    x_dot = A - x - y_dot
    return (x_dot, y_dot)

def scipy_brusselator(t, y):
    return np.array(brusselator(*y, A=A, B=B))

class BrusselatorParareal(FixedParareal, CachedPR):
    def __init__(self, a: float, b: float, n_coarse: int, n_fine: int, iterations: int, x_initial: np.ndarray):
        if self.get_cache(a, b, n_coarse, n_fine, iterations, x_initial):
            return
        super().__init__(a, b, n_coarse, n_fine, iterations, x_initial)
    
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        return RK4(brusselator, b-a, 2, x_in)[-1] # type: ignore
    
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        dt = t_vals[1] - t_vals[0]
        result = RK4(brusselator, dt, len(t_vals), x_in) # type: ignore
        if self.save_fine:
            return list(result)
        else:
            return [result[-1]]

def draw_plots2d(x, y, t, title):
    ax = plt.figure().subplots(1,1)
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    plt.show()
    
    ax = plt.figure().subplots(1,1)
    ax.plot(t, x, label='x')
    ax.plot(t, y, label='y')
    ax.set_xlabel('t')
    ax.set_title(title)
    plt.legend()
    
    plt.show()
    
def error_calc(x_coarse, comparison, t_coarse, plot_grid, title = None,
               save_name = None, *plot_args, **plot_kwargs):
    num_t, iterations, num_vars = x_coarse.shape
    assert num_vars == 2
    if plot_grid[0]*plot_grid[1] < iterations:
        raise ValueError('Grid is not big enough for number of iterations')
    
    errors = x_coarse - comparison[:, np.newaxis, :]
    error_distances = np.sqrt(np.sum(errors**2, axis=2))
            
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    axs = fig.subplots(*plot_grid, sharex=True, sharey=True).flatten()
            
    for i in range(iterations):
        axs[i].plot(t_coarse, error_distances[:, i], *plot_args, **plot_kwargs)
        
        axs[i].set_title(f'Iteration {i}')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('Error')
        
    if save_name:
        plt.savefig(f'tutorial/figs/brusselator/{save_name}')
    plt.show()
    
def main():
    time_steps_fine = 20
    time_steps_gross = 32
    time_range = [0, 12]
    initial_cond = np.array([0,1])
    dt = time_range[1]/(time_steps_fine*time_steps_gross)
    t = np.linspace(time_range[0], time_range[1], time_steps_fine*time_steps_gross+1)
    x, y = RK4(brusselator, dt, time_steps_fine*time_steps_gross+1, initial_cond, A=A, B=B).T
    
    draw_plots2d(x, y, t, 'Brusselator')
    
    pr_sol = BrusselatorParareal(time_range[0], time_range[1], time_steps_gross, time_steps_fine, 6, initial_cond)
    pr_sol.solve(processors=5, print_ref='Busselator', save_fine=True)
    pr_sol.save_cache()
    pr.plot_fine_comp(pr_sol.t_coarse, pr_sol.x_coarse_corr, pr_sol.t_fine, pr_sol.x_fine, ['x', 'y'], 'Brusselator')
    pr.plot_2d_phase(pr_sol.x_coarse_corr, ['x', 'y'], 'Brusselator', (x, y), 'brusselator')
    pr.plot_2d_phase_grid(pr_sol.x_coarse_corr, (2, 3), ['x', 'y'], 'Brusselator', (x, y))
    
    np_serial_sol = np.array([x,y]).T
    error_calc(pr_sol.x_coarse_corr, np_serial_sol[::time_steps_fine, :], pr_sol.t_coarse, (2, 3))
    animator = PRanimation2D(pr_sol.x_coarse_corr, pr_sol.x_fine, [[0,4], [0.5, 5]], ['x', 'y'], 10, 1,
                             title='Brusselator', line_colour=cm.get_cmap('YlOrRd_r'), dot_colour=cm.get_cmap('YlOrRd_r'))
    animator.animate('tutorial/animations/brusselator.gif', 10)

class AdaptiveBrusselator(BaseParareal):
    def __init__(self, a: float, b: float, coarse_step: float, iterations: int, x_initial: np.ndarray,
                 target_accuracy: float, classical_K: int):
        self.coarse_step_size = coarse_step
        n_coarse = int((b-a)/coarse_step)
        super().__init__(a, b, n_coarse, iterations, x_initial)
        self.target_eta = target_accuracy
        self.K = classical_K
        self.nfev = np.zeros((self.n_coarse, self.iterations))
    
    def get_fine_accuracy(self, k_iteration):
        if k_iteration < self.K:
            k_pow = (k_iteration+1)/self.K
            return self.coarse_step_size**(1-k_pow) * (self.target_eta/2)**k_pow
        else:
            return self.target_eta/2
        
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        sol = integ.solve_ivp(scipy_brusselator, (a,b), x_in)
        return sol.y[:, -1]
    
    def fine_integration(self, t_start: float, t_end: float, x_initial: np.ndarray,
                         coarse_step: int, iteration: int) -> Tuple[Optional[List[float]], List[np.ndarray]]:
        accuracy = self.get_fine_accuracy(iteration)
        sol = integ.solve_ivp(scipy_brusselator, (t_start, t_end), x_initial, 'Radau', dense_output=True, atol=accuracy, rtol=0)
        self._print(f'Iteration: {iteration}, step: {coarse_step}, evals:{sol.nfev}, accuracy:{accuracy}')
        
        self.nfev[coarse_step, iteration] = sol.nfev
        if self.save_fine:
            # Output as many points as calls made by scipy to give an indication of the number of fine steps
            t_fine = np.linspace(t_start, t_end, sol.nfev)
            return (list(t_fine), list(sol.sol(t_fine).T))
        else:
            return (None, [sol.sol(t_end)])
    
def adaptive_main():
    x0 = np.array([0,1])
    classic_iters_taken = 10
    eta = 1e-6
    # solve = AdaptiveBrusselator(0, 900, 0.1, 20, x0, eta, 20)
    solve = AdaptiveBrusselator(0, 12, 12/32, 10, x0, eta, classic_iters_taken)
    solve.solve(eta, 5)
    
    animator = PRanimationAdaptive2D(solve.x_coarse_corr, solve.x_fine, [[0,4], [0.5, 5]], ['x', 'y'], 10, 1,
                             title='Brusselator', line_colour=cm.get_cmap('YlOrRd_r'), dot_colour=cm.get_cmap('YlOrRd_r'))
    animator.animate('tutorial/animations/brusselator_adaptive.gif', 10)
    
if __name__ == '__main__':
    main()
    # adaptive_main()