from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from adaptive_parareal import FixedParareal, CachedPR
import burgers_funcs as b_funcs
import burgers
from burgers_q_funcs import *

BURGERS_INT_LIST: List[b_funcs.INT_FUNC_TYPE] = [b_funcs.burgers_imexRK, b_funcs.burgers_SL]
BURGERS_INT_NAMES: List[str] = ['IMEX', 'SL']

class BurgersParareal(FixedParareal, CachedPR):
    def __init__(self, t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                 x_range: Tuple[float, float], dx: float,
                 sol_func: Callable, q_func: Callable,
                 iterations: int, nu: float, coarse_int_func: b_funcs.INT_FUNC_TYPE):
        found_cache = self.get_cache(t_range, dt_fine, dt_coarse, x_range, dx, sol_func,
                                           q_func, iterations, nu, coarse_int_func)
        if found_cache: # Don't initialise if a cached object is being used instead
            return
        
        t_len = t_range[1] - t_range[0]
        n_coarse = int(t_len/dt_coarse)
        n_fine = int(t_len/dt_fine/n_coarse)
        n_x = int((x_range[1] - x_range[0])/dx)
        
        self.dx = dx
        self.dt_fine = dt_fine
        self.nu = nu
        self.coarse_int_func = coarse_int_func
        
        self.x_vals = np.linspace(x_range[0], x_range[1], n_x, endpoint=False)
        self.sol_func = sol_func
        self.q_func = q_func
        
        x_initial = self.sol_func(0, self.x_vals)
        super().__init__(t_range[0], t_range[1], n_coarse, n_fine, iterations, x_initial)
    
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        output: Union[List[np.ndarray], np.ndarray]
        if self.save_fine:
            output = np.empty((len(t_vals), *x_in.shape))
            output[0] = x_in
        else:
            output = [x_in]
        
        for i, t in enumerate(t_vals[:-1]):
            if i and i%200 == 0:
                self._print(f'Fine step {i}')
            output[(i+1)*self.save_fine] = b_funcs.burgers_imexRK(
                output[i*self.save_fine], self.dt_fine, self.dx, self.nu, None, self.x_vals, t, self.q_func)
            
        return list(output)
        
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        previous_step = coarse_step if coarse_step > 0 else 0
        u_nminus = self.x_coarse_corr[previous_step, iteration]
        return self.coarse_int_func(x_in, b-a, self.dx, self.nu, u_nminus, self.x_vals, a, self.q_func)
    
    def is_within_tol(self, x_current: np.ndarray, x_previous: np.ndarray) -> bool:
        max_diff = np.amax(np.abs(x_current - x_previous))
        self._print(f'Max difference: {max_diff: .3e}, Allowed tolerance: {self.tol}')
        return max_diff < self.tol
    
def serial_solve(x_vals, t_vals, nu, x0, q_func, integ_func: b_funcs.INT_FUNC_TYPE, print_info=False):
    dx = x_vals[1] - x_vals[0]
    dt = t_vals[1] - t_vals[0]
    
    output = np.empty((len(t_vals), len(x_vals)))
    output[0, :] = x0
    for i, t in enumerate(t_vals[:-1]):
        if i and i%200 == 0 and print_info:
            print(f'Step {i}')
        i_minus = i-1 if i > 0 else 0
        output[i+1, :] = integ_func(output[i, :], dt, dx, nu, output[i_minus, :], x_vals, t, q_func)
        
    return output

def serial_solves(x_range, t_range, dx_vals, dt_vals, nu_vals, integ_func: b_funcs.INT_FUNC_TYPE,
                  sol_func: Callable, q_func: Callable, print_error=False, plots=False):
    """Reproduce table 2 from Schmitt et al."""
    total_width = 8*len(dx_vals) + 7
    for dt in dt_vals:
        text = f'dt = {dt:.0e}'
        print(f'{text:<{total_width}s}  ', end='')
    print()
    print(('-'*total_width + '  ')*len(dt_vals))
    for _ in dt_vals:
        text = "nu\\N"
        print(f'{text:<7s}', end='')
        for dx in dx_vals:
            text = f'1/{1/dx:.0f}'
            print(f'{text:^8s}', end='')
        print('  ', end='')
    print()
    for nu in nu_vals:
        for dt in dt_vals:
            n_t = int((t_range[1] - t_range[0])/dt)
            t_vals = np.linspace(t_range[0], t_range[1], n_t+1)
            print(f'{nu:<7.0e}', end='')
            for dx in dx_vals:
                n_x = int((x_range[1] - x_range[0])/dx)
                x_vals = np.linspace(x_range[0], x_range[1], n_x+1)
                x_initial = sol_func(0, x_vals)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    sol = serial_solve(x_vals, t_vals, nu, x_initial, q_func, integ_func)
                    if plots:
                        burgers.plot_burgers(t_vals, x_vals, sol, f'dt: {dt:.0e}, dx: 1/{1/dx:.0f}, nu: {nu:.0e}', zlim=[-1, 1])
                    if np.isnan(sol).any():
                        print(f'{"x":^8s}', end='')
                    else:
                        x_grid, t_grid = np.meshgrid(x_vals, t_vals)
                        true_sol = sol_func(t_grid, x_grid)
                        if print_error:
                            max_err = np.amax(np.abs(sol-true_sol))
                            print(f'{max_err:^8.1e}', end='')
                        else:
                            print(f'{"o":^8s}', end='')
            print('  ', end='')
        print()
        
def get_max_error(u, u_true):
    return np.amax(np.abs(u - u_true), axis=1)

def step_scheme_errors(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                       x_range: Tuple[float, float], dx: float,
                       sol_func: Callable, q_func: Callable,
                       iterations: int, nu: float, tol: Optional[float]):
    """Reproduce fig. 4 from Schmitt et al."""
    fig = plt.figure()
    axs: List[Axes] = fig.subplots(1, len(BURGERS_INT_LIST))
    if len(BURGERS_INT_LIST) == 1:
        axs = [axs]
    fine_sol_imex = None
    for i, coarse_int_func in enumerate(BURGERS_INT_LIST):
        name = BURGERS_INT_NAMES[i]
        print(f'Starting {name}')
        sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func, iterations, nu, coarse_int_func)
        
        # On the first iteration find the true to solution for error calculation
        if i == 0:
            print(f'{name}:', 'Calculating true solution')
            x_grid, t_grid = np.meshgrid(sol.x_vals, sol.t_coarse)
            true_sol = sol_func(t_grid, x_grid)

        print(f'{name}:', 'Starting parareal solve')
        iters_done = sol.solve(tolerance=tol, processors=5, print_ref=name, save_fine=False)
        sol.save_cache()
        for k in range(iters_done+1):
            axs[i].plot(get_max_error(sol.x_coarse_corr[:, k, :], true_sol), label=f'k = {k}')
            
        # Calculate fine solution
        print(f'{name}:', 'Starting fine solve')
        fine_t_vals = np.linspace(t_range[0], t_range[1], sol.n_coarse*sol.n_fine+1)
        fine_sol = serial_solve(sol.x_vals, fine_t_vals, nu, sol.sol_func(0, sol.x_vals), sol.q_func, coarse_int_func, True)
        fine_sol_coarse = fine_sol[::sol.n_fine, :]
        axs[i].plot(get_max_error(fine_sol_coarse, true_sol), '--', label='$\mathcal{F}_{'+name+'}$')
        # Save the IMEX solution
        if name == 'IMEX' and fine_sol_imex is None:
            fine_sol_imex = fine_sol_coarse.copy()
        # Also plot the fine IMEX solution on the SL plot
        if name == 'SL':
            if fine_sol_imex is None:
                print(f'{name}:', 'Starting IMEX fine solve')
                fine_t_vals = np.linspace(t_range[0], t_range[1], sol.n_coarse*sol.n_fine+1)
                fine_sol = serial_solve(sol.x_vals, fine_t_vals, nu, sol.sol_func(0, sol.x_vals), sol.q_func, burgers.burgers_imexRK, True)
                fine_sol_imex = fine_sol[::sol.n_fine, :]
            axs[i].plot(get_max_error(fine_sol_imex, true_sol), '--', label='$\mathcal{F}_{IMEX}$')
            
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Coarse time step')
        axs[i].set_ylabel('$\epsilon_{max}$')
        axs[i].set_ylim(top=10)
        axs[i].set_title(f'Coarse solver: {name}')
        axs[i].legend()
    fig.suptitle(f'Errors for different coarse solvers\n$\Delta t = {dt_fine}, \Delta T = {dt_coarse}, \Delta x = 1/{1/dx:.0f}$')
    plt.show()
    
def nu_errors(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
              x_range: Tuple[float, float], dx: float,
              sol_func: Callable, q_func: Callable,
              iterations: int, nu_vals: Iterable[float],
              tol: Optional[float], coarse_func: b_funcs.INT_FUNC_TYPE,
              plot_iters: Sequence[int]):
    """Reproduce fig. 5,6 from Schmitt et al."""
    fig = plt.figure()
    axs: List[Axes] = fig.subplots(len(plot_iters),1)
    if len(plot_iters) == 1:
        axs = [axs]
    for i, nu in enumerate(nu_vals):
        print(f'Starting nu={nu}')
        sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func, iterations, nu, coarse_func)
        if i == 0:
            print(f'nu={nu}:', 'Calculating true solution')
            x_grid, t_grid = np.meshgrid(sol.x_vals, sol.t_coarse)
            true_sol = sol_func(t_grid, x_grid)
            
        print(f'nu={nu}:', 'Starting parareal solve')
        iters_done = sol.solve(tolerance=tol, processors=5, print_ref=f'nu={nu}', save_fine=False)
        sol.save_cache()
        for j, k in enumerate(plot_iters):
            # Check the solve didn't stop before reaching this iteration
            if k > iters_done:
                continue
            axs[j].plot(get_max_error(sol.x_coarse_corr[:, k, :], true_sol), label=f'nu = {nu}')
            
    for i, iter_num in enumerate(plot_iters):
        axs[i].set_title(f'Iteration {iter_num}')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Coarse time step')
        axs[i].set_ylabel('$\epsilon_{max}$')
        axs[i].set_ylim(top=10)
        axs[i].legend()
    fig.suptitle(f'Errors for different values of $\nu$\n$\Delta t = {dt_fine}, \Delta T = {dt_coarse}, \Delta x = 1/{1/dx:.0f}$')
    plt.show()
    
def time_step_convergence(t_range: Tuple[float, float],
                          dt_fine: float, dt_coarse_vals: Iterable[float],
                          x_range: Tuple[float, float], dx: float,
                          sol_func: Callable, q_func: Callable,
                          max_iterations: int, nu_vals: Iterable[float], tol: float,
                          coarse_int_funcs: Iterable[Tuple[b_funcs.INT_FUNC_TYPE, str]]):
    """Reproduce table 3 from Schmitt et al."""
    output = ''
    output += f'{"nu":<6s} {"dT":<10s}  '
    for _, func_name in coarse_int_funcs:
        text = f'C: {func_name:<5s}'
        output += f'{text:<8s}  '
    output += '\n'
    for nu in nu_vals:
        for dt_coarse in dt_coarse_vals:
            output += f'{nu:<6} {dt_coarse:<10.2e}  '
            for coarse_func, func_name in coarse_int_funcs:
                sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func, max_iterations, nu, coarse_func)
                iters_taken = sol.solve(tol, 5, func_name, False)
                sol.save_cache()
                num_text = f'{iters_taken:>3d}'
                output += f'{num_text:<8s}  '
            output += '\n'
            print(output)
    print(output)
    
def nu_convergence(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                   x_range: Tuple[float, float], dx: float,
                   sol_func: Callable, q_func: Callable,
                   max_iterations: int, nu_vals: Iterable[float], tol: float,
                   coarse_int_funcs: Iterable[Tuple[b_funcs.INT_FUNC_TYPE, str]]):
    """Reproduce figs. 8 from Schmitt et al."""
    fig = plt.figure()
    ax: Axes = fig.subplots(1,1)
    for coarse_func, func_name in coarse_int_funcs:
        iters_taken = []
        for nu in nu_vals:
            sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func, max_iterations, nu, coarse_func)
            iters_taken.append(sol.solve(tol, 5, f'{func_name}, nu={nu}', False))
            sol.save_cache()
            # for i in range(1, iters_taken[-1]+1):
            #     burgers.plot_burgers(sol.t_coarse, sol.x_vals, sol.x_coarse_corr[:, i, :], f'Iteration {i}')
        ax.plot(nu_vals, iters_taken, 'o-', label='$\mathcal{C}_{'+func_name+'}$')
    ax.set_xlabel('$\\nu$')
    ax.set_ylabel('k')
    ax.set_xscale('log')
    ax.set_title(f'Iterations to converge\n$\Delta t = {dt_fine}, \Delta T = {dt_coarse}, \Delta x = 1/{1/dx:.0f}$')
    plt.legend()
    plt.show()
    
def iteration_error(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                    x_range: Tuple[float, float], dx: float, sol_func: Callable,
                    q_func: Callable, coarse_int_func: b_funcs.INT_FUNC_TYPE, max_iterations: int,
                    nu: float, tol: Optional[float], name: Optional[str] = None):
    """Does a parareal solve and returns the error after each iteration"""
    sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func,
                          max_iterations, nu, coarse_int_func)
    iters_complete = sol.solve(tol, 5, name, False)
    sol.save_cache()
    errors = np.abs(sol.x_coarse_corr[:, 1:iters_complete+1, :] - sol.x_coarse_corr[:, 0:iters_complete, :])
    error_lst = []
    for iteration in errors.swapaxes(0, 1):
        error_lst.append(np.amax(iteration))
        
    return np.array(error_lst)

def plot_error_change(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                      x_range: Tuple[float, float], dx: float,
                      sol_func: Callable, q_func: Callable,
                      max_iterations: int, nu_vals: Iterable[float], tol: Optional[float],
                      coarse_int_funcs: Sequence[Tuple[b_funcs.INT_FUNC_TYPE, str]]):
    """Reproduce fig. 7 from Schmitt et al."""
    fig = plt.figure()
    axs: List[Axes] = fig.subplots(1, len(coarse_int_funcs))
    if len(coarse_int_funcs) == 1:
        axs = [axs]
    for i, (coarse_int_func, name) in enumerate(coarse_int_funcs):
        for nu in nu_vals:
            error_vals = iteration_error(t_range, dt_fine, dt_coarse, x_range, dx, sol_func,
                                         q_func, coarse_int_func, max_iterations, nu, tol, f'{name}, nu={nu}')
            axs[i].plot(error_vals, label=f'$\\nu=${nu}')
            
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel('$\epsilon_{max}$')
        axs[i].set_ylim(top=10)
        axs[i].set_title(f'Coarse function: {name}\n$\Delta t = {dt_fine}, \Delta T = {dt_coarse}, \Delta x = 1/{1/dx:.0f}$')
        axs[i].legend()
    plt.show()
    
    
def benchmark1():
    K = 3
    x_range = (0.,1.)
    t_range = (0.,1.)
    Q_func = partial(B1_q, K)
    sol_func = partial(B1_sol, K)
    
    
    # serial_solves(x_range, t_range, [1/64, 1/128, 1/256], [1e-4, 1e-3, 1e-2], [0]+[10**x for x in range(-4,1)], b_funcs.burgers_SL, sol_func, Q_func, True)
    
    # B1 parameters
    B1_dt_fine = 1e-5 # 1e-6
    B1_dt_coarse = 1e-2
    B1_dx = 1/256
    B1_tolerance = 1e-8
    
    # step_scheme_errors(t_range, 1e-5, B1_dt_coarse, x_range, B1_dx, sol_func, Q_func, 5, 1e-2, B1_tolerance)
    B1_nu_vals = 1e-3*np.array([1, 2, 4, 6, 8, 10])
    # nu_errors(t_range, B1_dt_fine, B1_dt_coarse, x_range, B1_dx, sol_func, Q_func, 81, nu_vals, B1_tolerance,
    #           burgers.burgers_imexRK, [1, 2, 40, 80])
    nu_errors(t_range, B1_dt_fine, B1_dt_coarse, x_range, B1_dx, sol_func, Q_func, 3, B1_nu_vals, B1_tolerance,
              burgers.burgers_SL, [1, 2])
    
    B1_dt_coarse_vals = [2.5e-3, 1e-3]
    # time_step_convergence(t_range, B1_dt_fine, B1_dt_coarse_vals, x_range, B1_dx, sol_func, Q_func, 8,
    #                       [0.005, 0.01], 1e-6, ((burgers.burgers_imexRK, 'IMEX'), (burgers.burgers_SL, 'SL')))
    
def benchmark2():
    K_MAX = 3
    EPSILON = 0.1
    x_range = (0.,1.)
    t_range = (0.,1.)
    sol_func = partial(B2_sol, K_MAX, EPSILON)
    Q_func = partial(B2_q, K_MAX, EPSILON)
    
    # serial_solves(x_range, t_range, [1/64], [1e-3], [1e-4], burgers.burgers_imexRK, sol_func, Q_func, plots=True)
    
    B2_dt_fine = 1e-5 # 1e-6
    B2_dt_coarse = 1e-2
    B2_dx = 1/256
    B2_tolerance = 1e-6
    
    nu_vals = [0.0001*x for x in range(1, 10)] + [0.001*x for x in range(1, 11)]
    # nu_convergence(t_range, B2_dt_fine, B2_dt_coarse, x_range, B2_dx, sol_func, Q_func,
    #                20, nu_vals, B2_tolerance, ((burgers.burgers_SL, 'SL'), (burgers.burgers_imexRK, 'IMEX')))
    plot_error_change(t_range, B2_dt_fine, B2_dt_coarse, x_range, B2_dx, sol_func, Q_func,
                      10, [0]+[10**x for x in range(-4,1)], B2_tolerance, ((burgers.burgers_SL, 'SL'), (burgers.burgers_imexRK, 'IMEX')))
    
if __name__ == '__main__':
    # benchmark1()
    benchmark2()
    