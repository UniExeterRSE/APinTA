from functools import partial
from typing import Callable, List, Tuple
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes

from adaptive_parareal import FixedParareal
import burgers_funcs as b_funcs
import burgers
from burgers_q_funcs import *

# BURGERS_INT_LIST: List[b_funcs.INT_FUNC_TYPE] = [b_funcs.burgers_imexRK]
# BURGERS_INT_NAMES: List[str] = ['IMEX']
BURGERS_INT_LIST: List[b_funcs.INT_FUNC_TYPE] = [b_funcs.burgers_explicitRK, b_funcs.burgers_imexRK, b_funcs.burgers_SL]
BURGERS_INT_NAMES: List[str] = ['RK', 'IMEX', 'SL']
# BURGERS_INT_LIST: List[b_funcs.INT_FUNC_TYPE] = [b_funcs.burgers_imexRK, b_funcs.burgers_SL]
# BURGERS_INT_NAMES: List[str] = ['IMEX', 'SL']

class BurgersParareal(FixedParareal):
    def __init__(self, t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                 x_range: Tuple[float, float], dx: float,
                 sol_func: Callable, q_func: Callable,
                 iterations: int, nu: float, coarse_int_func: b_funcs.INT_FUNC_TYPE):
        t_len = t_range[1] - t_range[0]
        n_coarse = int(t_len/dt_coarse)
        n_fine = int(t_len/dt_fine/n_coarse)
        n_x = int((x_range[1] - x_range[0])/dx)
        
        self.dx = dx
        self.dt_fine = dt_fine
        self.nu = nu
        self.coarse_int_func = coarse_int_func
        
        self.x_vals = np.linspace(x_range[0], x_range[1], n_x+1)
        self.sol_func = sol_func
        self.q_func = q_func
        
        x_initial = self.sol_func(0, self.x_vals)
        super().__init__(t_range[0], t_range[1], n_coarse, n_fine, iterations, x_initial)
    
    def fine_integration_func(self, t_vals: List[float], x_in: np.ndarray) -> List[np.ndarray]:
        output = np.empty((len(t_vals), *x_in.shape))
        output[0] = x_in
        
        for i, t in enumerate(t_vals[:-1]):
            if i and i%200 == 0:
                self.print(f'Fine step {i}')
            output[i+1] = b_funcs.burgers_imexRK(output[i], self.dt_fine, self.dx, self.nu, None, self.x_vals, t, self.q_func)
            
        return list(output)
        
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray, coarse_step: int, iteration: int) -> np.ndarray:
        previous_step = coarse_step if coarse_step > 0 else 0
        u_nminus = self.x_coarse[previous_step, iteration]
        return self.coarse_int_func(x_in, b-a, self.dx, self.nu, None, self.x_vals, a, self.q_func)
    
def serial_solve(x_vals, t_vals, nu, x0, q_func, integ_func: b_funcs.INT_FUNC_TYPE):
    dx = x_vals[1] - x_vals[0]
    dt = t_vals[1] - t_vals[0]
    
    output = np.empty((len(t_vals), len(x_vals)))
    output[0, :] = x0
    for i, t in enumerate(t_vals[:-1]):
        i_minus = i-1 if i > 0 else 0
        output[i+1, :] = integ_func(output[i, :], dt, dx, nu, None, x_vals, t, q_func)
        
    return output

def serial_solves(x_range, t_range, dx_vals, dt_vals, nu_vals, integ_func: b_funcs.INT_FUNC_TYPE,
                  sol_func: Callable, q_func: Callable, print_error=False, plots=False):
    total_width = 7*len(dx_vals) + 7
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
            print(f'{text:^7s}', end='')
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
                        burgers.plot_burgers(t_vals, x_vals, sol, f'dt: {dt:.0e}, dx: 1/{1/dx:.0f}, nu: {nu:.0e}')
                    if np.isnan(sol).any():
                        print(f'{"x":^7s}', end='')
                    else:
                        x_grid, t_grid = np.meshgrid(x_vals, t_vals)
                        true_sol = sol_func(t_grid, x_grid)
                        if print_error:
                            max_err = np.amax(np.abs(sol-true_sol))
                            print(f'{max_err:^7.0e}', end='')
                        else:
                            print(f'{"o":^7s}', end='')
            print('  ', end='')
        print()
        
def get_max_error(u, u_true):
    return np.amax(np.abs(u - u_true), axis=1)

def step_scheme_errors(t_range: Tuple[float, float], dt_fine: float, dt_coarse: float,
                 x_range: Tuple[float, float], dx: float,
                 sol_func: Callable, q_func: Callable,
                 iterations: int, nu: float):
    fig = plt.figure()
    axs: List[Axes] = fig.subplots(len(BURGERS_INT_LIST),1)
    if len(BURGERS_INT_LIST) == 1:
        axs = [axs]
    for i, coarse_int_func in enumerate(BURGERS_INT_LIST):
        print(f'Starting {BURGERS_INT_NAMES[i]}')
        sol = BurgersParareal(t_range, dt_fine, dt_coarse, x_range, dx, sol_func, q_func, iterations, nu, coarse_int_func)
        if i == 0:
            print('Calculating true solution')
            x_grid, t_grid = np.meshgrid(sol.x_vals, sol.t_coarse)
            true_sol = sol_func(t_grid, x_grid)
            
        print('Starting fine solve')
        fine_t_vals = np.linspace(t_range[0], t_range[1], sol.n_coarse*sol.n_fine+1)
        fine_sol = serial_solve(sol.x_vals, fine_t_vals, nu, sol.sol_func(0, sol.x_vals), sol.q_func, BURGERS_INT_LIST[i])
        fine_sol_coarse = fine_sol[::sol.n_fine, :]
        axs[i].plot(get_max_error(fine_sol_coarse, true_sol), '--', label='$\mathcal{F}_{'+BURGERS_INT_NAMES[i]+'}$')
        
        print('Starting parareal solve')
        sol.solve(5, True)
        for k in range(iterations):
            axs[i].plot(get_max_error(sol.x_coarse[:, k, :], true_sol), label=f'k = {k+1}')
            # burgers.plot_burgers(sol.t_coarse, sol.x_vals, sol.x_coarse[:, k, :] - true_sol, f'k = {k+1}', zlim=None)
            # burgers.plot_burgers(sol.t_coarse, sol.x_vals, sol.x_coarse[:, k, :] - fine_sol_coarse, f'k = {k+1}', zlim=None)
            
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Coarse time step')
        axs[i].set_ylabel('$\epsilon_{max}$')
        axs[i].set_ylim(top=10)
        axs[i].legend()
    plt.show()
        
    
if __name__ == '__main__':
    K = 3
    
    x_range = (0.,1.)
    t_range = (0.,1.)
    
    B1_q = partial(B1_q, K)
    B1_sol = partial(B1_sol, K)
    
    # serial_solves(x_range, t_range, [1/64, 1/128, 1/256], [1e-4, 1e-3, 1e-2], [0]+[10**x for x in range(-4,1)], b_funcs.burgers_SL, B1_sol, B1_q, True)
    step_scheme_errors(t_range, 1e-4, 1e-2, x_range, 1/256, B1_sol, B1_q, 4, 1e-2)