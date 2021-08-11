from typing import Callable, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

# Number of processes to use when running fine integration in parallel. Runs in serial if set to 0
PARALLEL_PROCESSES = 5

def parareal(a: float, b: float, n_gross: int, n_fine: int, iterations: int, x_initial: Sequence[float],
             coarse_integ: Callable, fine_integ: Callable, func: Optional[Callable] = None,
             integ_args = (), full_output: bool = False, **integ_kwargs):
    """
    a : float
        start of integration
    b : float
        end of integration
    n_gross : int
        number of gross steps
    n_fine : int
        number fo fine steps within each gross step
    iterations : int
        number of parallel iterations to be done
    x0 : Sequence[float]
        Initial conditions
    func : (*variables, **func_kwargs) -> variables_dot or None
        function to be integrated over. Returns the time derivate of each variable
    coarse_integ/fine_integ : (func, dt, *integ_args, n, x0, **integ_kwargs) -> np.array or
                              (dt, *integ_args, n, x0, **integ_kwargs) -> np.array if func = None
        Integrates the function coarsely/finely respectively. Given an x0 array of length M, must
        return an array of length M x n.
    integ_args : Tuple
        Arguments to be passed to the integrating functions
    full_output : bool = False
        Whether to return the fine results as well as the coarse results
    **integ_kwargs : keyword arguments
        Keyword arguments to be passed to the integrating functions
    """
    n_vars = len(x_initial)
    x0 = np.array(x_initial)
    # Add 1 to n_fine as we always want it with one extra to include the endpoint
    n_fine += 1
    
    t_gross = np.linspace(a, b, n_gross+1)
    t_fine = np.empty((n_gross, n_fine))
    for i in range(n_gross):
        left, right = t_gross[i], t_gross[i+1]
        t_fine[i, :] = np.linspace(left, right, n_fine)
    dt_gross = (b - a)/n_gross
    dt_fine = t_fine[0,1] - t_fine[0,0]
    
    # Creates a function that does the fine integration requiring only the initial conditions
    # as a parameter
    if func is None:
        fine_int_func = partial(fine_integ, dt_fine, *integ_args, n_fine, **integ_kwargs)
        coarse_int_func = partial(coarse_integ, dt_gross, *integ_args, **integ_kwargs)
    else:
        fine_int_func = partial(fine_integ, func, dt_fine, *integ_args, n_fine, **integ_kwargs)
        coarse_int_func = partial(coarse_integ, func, dt_gross, *integ_args, **integ_kwargs)
    
    x_gross = np.empty((len(t_gross), iterations, n_vars))
    x_fine_corr = np.empty((n_gross, iterations, n_fine, n_vars))
    
    # Add initial conditions
    x0_repeated = x0.reshape((1, n_vars,)).repeat(iterations, 0)
    x_gross[0, :] = x0_repeated
    x_fine_corr[0, :, 0] = x0_repeated
    
    # Initial coarse integration
    x_gross[:, 0] = coarse_int_func(n_gross+1, x_gross[0, 0])    
    
    x_gross_corr = x_gross.copy()
    
    for k in range(1, iterations):
        print(f'Iteration {k}')
        x_fine_corr[:, k, 0] = x_gross_corr[:-1, k-1]
        if PARALLEL_PROCESSES == 0:
            # Loop done in serial
            for i in range(n_gross):
                x_fine_corr[i, k, :] = fine_int_func(x_fine_corr[i, k, 0])
        else:
            # Loop done in parallel
            with Pool(PARALLEL_PROCESSES) as p:
                integ_map = p.map(fine_int_func, x_fine_corr[:, k, 0])
            x_fine_corr[:, k, :] = np.array(list(integ_map))
            
        # Correcting
        for t in range(n_gross):
            x_gross[t+1, k] = coarse_int_func(2, x_gross_corr[t, k])[-1]
            x_gross_corr[t+1, k] = x_gross[t+1, k] - x_gross[t+1, k-1] + x_fine_corr[t, k, -1]
          
    if full_output:
        return (t_gross, x_gross_corr, t_fine, x_fine_corr)
    else:
        return (t_gross, x_gross_corr)
    
def plot_comp(t_gross, x_gross, x_fine, var_names = None, title = None,
              *plot_args, **plot_kwargs):
    x_gross = x_gross.swapaxes(1,2).swapaxes(0,1)
    x_fine = x_fine.swapaxes(0,1).swapaxes(0,3)
    num_vars, num_t, iterations = x_gross.shape
    if var_names is None:
        var_names = []
        for x in range(num_vars):
            var_names[x] = f'Variable {x}'
    # Include the initial conditions in the x_fine array
    x_fine_adj = np.concatenate((x_fine[:, 0, 0, :].reshape(num_vars, 1, iterations), x_fine[:, :, -1, :]), axis=1)
            
    for i in range(iterations):
        fig = plt.figure(figsize=(10,8))
        if title is None:
            fig.suptitle(f'Iteration {i}')
        else:
            fig.suptitle(f'{title}: Iteration {i}')
        axs = fig.subplots(num_vars, 1)
        for x in range(num_vars):
            axs[x].plot(t_gross, x_gross[x, :, i], label=var_names[x], *plot_args, **plot_kwargs)
            axs[x].plot(t_gross, x_fine_adj[x, :, i], label=var_names[x]+' corr', *plot_args, **plot_kwargs)
            axs[x].set_ylabel(var_names[x])
            axs[x].legend()
        axs[-1].set_xlabel('Time')
        plt.show()
        
def plot_fine_comp(t_gross, x_gross, t_fine, x_fine, var_names = None, title = None,
              *plot_args, **plot_kwargs):
    num_gross, iterations = x_fine.shape
    num_vars = len(x_fine[-1, -1][-1])
    if var_names is None:
        var_names = []
        for x in range(num_vars):
            var_names.append(f'Variable {x}')
            
    for i in range(1, iterations):
        fig = plt.figure(figsize=(10,8))
        if title is None:
            fig.suptitle(f'Iteration {i}')
        else:
            fig.suptitle(f'{title}: Iteration {i}')
        axs = fig.subplots(num_vars, 1)
        for x in range(num_vars):
            axs[x].plot(t_gross, x_gross[:, i-1, x], '-o', *plot_args, **plot_kwargs)
            for f in range(num_gross):
                fine_vals = list(val_lst[x] for val_lst in x_fine[f, i])
                axs[x].plot(t_fine[f, i], fine_vals, *plot_args, **plot_kwargs)
                axs[x].set_ylabel(var_names[x])
        axs[-1].set_xlabel('Time')
        plt.show()
        
def plot_2d_phase(x_gross, var_names = None, title = None, comparison = None,
                  save_name = None, *plot_args, **plot_kwargs):
    num_t, iterations, num_vars = x_gross.shape
    assert num_vars == 2
    if var_names is None:
        var_names = []
        for x in range(num_vars):
            var_names.append(f'Variable {x}')
            
    for i in range(iterations):
        fig = plt.figure(figsize=(10,8))
        if title is None:
            fig.suptitle(f'Iteration {i}')
        else:
            fig.suptitle(f'{title}: Iteration {i}')
        axs = fig.subplots(1, 1)
        if comparison is not None:
            axs.plot(comparison[0], comparison[1], *plot_args, **plot_kwargs)
        axs.plot(x_gross[:, i, 0], x_gross[:, i, 1], 'o', *plot_args, **plot_kwargs)
        
        axs.set_xlabel(var_names[0])
        axs.set_ylabel(var_names[1])
        if save_name:
            plt.savefig(f'tutorial/figs/brusselator/{save_name}_{i}')
        plt.show()
        
def plot_2d_phase_grid(x_coarse, grid_shape, var_names = None, title = None,
                       comparison = None, save_name = None, *plot_args, **plot_kwargs):
    num_t, iterations, num_vars = x_coarse.shape
    assert num_vars == 2
    if grid_shape[0]*grid_shape[1] < iterations:
        raise ValueError('Grid is not big enough for number of iterations')
    if var_names is None:
        var_names = []
        for x in range(num_vars):
            var_names.append(f'Variable {x}')
            
    fig = plt.figure()
    if title:
        fig.suptitle(title)
    axs = fig.subplots(*grid_shape, sharex=True, sharey=True).flatten()
            
    for i in range(iterations):
        if comparison is not None:
            axs[i].plot(comparison[0], comparison[1], *plot_args, **plot_kwargs)
        axs[i].plot(x_coarse[:, i, 0], x_coarse[:, i, 1], 'o', *plot_args, **plot_kwargs)
        
        axs[i].set_title(f'Iteration {i}')
        axs[i].set_xlabel(var_names[0])
        axs[i].set_ylabel(var_names[1])
        
    if save_name:
        plt.savefig(f'tutorial/figs/brusselator/{save_name}')
    plt.show()
    