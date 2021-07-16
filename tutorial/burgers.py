from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as optimize

import parareal as pr

NU = 0.02
PLOT_ITERATION = False

def burgers_f(u, dx, nu, extra_out=False):
    u_iplus = np.roll(u, -1)
    u_iminus = np.roll(u, 1)
    result = -u*(u_iplus - u_iminus)/(2*dx) + nu*(u_iplus - 2*u + u_iminus)/dx**2
    result[0] = 0
    result[-1] = 0
    
    if extra_out:
        return (result, u_iplus, u_iminus)
    return result

def initial_func(x):
    return np.sin(2*np.pi*x)

def burgers_scipy_func(u_nplus, u_n, dt, dx, nu):
    zero = u_nplus - u_n - dt*burgers_f(u_nplus, dx, nu)
    return zero

def burgers_scipy(u_n, dt, dx, nu=0.02, **_):
    u_nplus = optimize.fsolve(burgers_scipy_func, u_n, (u_n, dt, dx, nu))
    burgers_scipy_func(u_nplus, u_n, dt, dx, nu)
    return u_nplus

def burgers_fixed_point(u_n, dt, dx, nu=0.02, tol=1e-5, max_iterations=10, x_vals=None, t=None):
    u_k = u_n
    if PLOT_ITERATION:
        fig = plt.figure()
        fig.suptitle(f'Time: {t}')
        ax = fig.subplots(max_iterations+1, 1)
    for i in range(max_iterations):
        u_kminus = u_k
        if PLOT_ITERATION:
            f, plus, minus = burgers_f(u_k, dx, nu, True)
            ax[i].plot(x_vals, u_k)
            ax[i].plot(x_vals, plus - minus)
            ax[i].plot(x_vals, plus - 2*u_k + minus)
        else:
            f = burgers_f(u_k, dx, nu)
        u_k = u_n + dt*f
        if max(abs(u_k-u_kminus)) < tol:
            break
    if PLOT_ITERATION:
        ax[max_iterations].plot(x_vals, u_k)
        plt.show()
    return u_k

def integ_burgers(dt, dx, n, u0, nu=0.02, tol=1e-5, max_iterations=10):
    u_len = len(u0)
    output = np.empty((u_len, n))
    output[:, 0] = u0
    
    for i in range(1, n):
        output[:, i] = burgers_scipy(output[:, i-1], dt, dx, nu)
        
    return output

def solve_burgers(t_range : Tuple[float, float], x_vals, num_t, x0, nu):
    t_vals = np.linspace(*t_range, num_t, False)
    num_x = len(x_vals)
    u_vals = np.zeros((num_t, num_x))
    dt = (t_range[1] - t_range[0])/num_t
    dx = x_vals[1] - x_vals[0]
    
    u_vals[0, :] = x0
    
    for i in range(1, num_t):
        u_vals[i, :] = burgers_scipy(u_vals[i-1, :], dt, dx, nu, x_vals=x_vals, t=t_vals[i])
        
    return t_vals, u_vals

def join_fine(t_fine, u_fine):
    num_vars, n_gross, n_fine, iterations = u_fine.shape
    n_fine -= 1 # Don't includ eoverlapping endpoints
    t_joined = np.empty(n_gross*n_fine)
    u_joined = np.empty((num_vars, n_gross*n_fine, iterations))
    
    for i in range(n_gross):
        t_joined[n_fine*i : n_fine*(i+1)] = t_fine[i, :-1]
        u_joined[:, n_fine*i : n_fine*(i+1), :] = u_fine[:, i, :-1, :]
        
    return (t_joined, u_joined)
    
def plot_burgers(t, x, u, title=None, save_name=None):
    shape0, shape1 = u.shape
    plot_burgers_fine(t.reshape(1, shape0), x, u.reshape(shape0, 1, shape1), title, save_name)
    
def plot_burgers_fine(t_fine, x, u_fine, title=None, save_name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    for i in range(u_fine.shape[1]):
        x_grid, t_grid = np.meshgrid(x, t_fine[i, :])
        surface = ax.plot_surface(x_grid, t_grid, u_fine[:, i, :], cmap=cm.turbo, antialiased=False, vmin=-1, vmax=1)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    if title:
        ax.set_title(title)
    
    fig.colorbar(surface)
    if save_name:
        plt.savefig(f"tutorial/figs/burgers/{save_name}.png")
    
    plt.show()
    
def find_errors(u_para, u_true):
    n_t, n_x, iterations = u_para.shape
    diff = u_para - u_true.reshape((n_t, n_x, 1)).repeat(iterations, 2)
    diff_squared = diff**2
    true_squared = u_true**2
    l2_error = np.sqrt(np.sum(diff_squared, 1))
    l2_norm = np.sqrt(np.sum(true_squared, 1))
    l2_norm_error = l2_error.T/l2_norm
    linf_error = np.amax(l2_norm_error, 1)/np.amax(l2_norm)
    
    return linf_error

def plot_errors(vals, labels):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    iterations = len(vals[0])
    for i in range(len(vals)):
        ax.plot(range(iterations), vals[i], '.-', label=labels[i])
        print(vals[i])
        
    ax.set_xlim([0, iterations])
    ax.set_xlabel('n')
    ax.set_ylabel('errors')
    ax.set_yscale('log')
    plt.legend()
    plt.show()
    
def errors_tmax(t_max_vals, x_values, x0, n_coarse, n_fine, iterations):
    dx = x_values[1] - x_values[0]
    error_lst = []
    label_lst = []
    
    for t in t_max_vals:
        t_vals, u_vals = solve_burgers((0,t), x_values, n_fine*n_coarse, x0, NU)
        *_, t_fine, u_fine = pr.parareal(0, t, n_coarse, n_fine, iterations, x0, integ_burgers,
                                         integ_burgers, integ_args=(dx,), nu=NU, full_output=True)
        para_t, para_u = join_fine(t_fine, u_fine)
        para_u = para_u.swapaxes(0,1)

        error = find_errors(para_u, u_vals)
        error_lst.append(error)
        label_lst.append(f'T={t}')
        
    plot_errors(error_lst, label_lst)
    
def errors_discretisation(dx_div_vals, n_fine_mult_vals, dx0, n_fine0, x_range, x0_func, n_coarse, t_max, iterations):
    error_lst = []
    label_lst = []
    
    for dx_div, n_fine_mult in zip(dx_div_vals, n_fine_mult_vals):
        dx = dx0/dx_div
        n_fine = n_fine0*n_fine_mult
        x_vals = np.arange(x_range[0], x_range[1]+dx, dx)
        x_initial = x0_func(x_vals)
        
        t_vals, u_vals, = solve_burgers((0, t_max), x_vals, n_fine*n_coarse, x_initial, NU)
        *_, t_fine, u_fine = pr.parareal(0, t_max, n_coarse, n_fine, iterations, x_initial, integ_burgers,
                                         integ_burgers, integ_args=(dx,), nu=NU, full_output=True)
        para_t, para_u = join_fine(t_fine, u_fine)
        para_u = para_u.swapaxes(0,1)

        error = find_errors(para_u, u_vals)
        error_lst.append(error)
        label_lst.append(f'dx=dx/{dx_div}, dt=dt/{n_fine_mult}')
        
    plot_errors(error_lst, label_lst)
    
if __name__ == '__main__':
    x_range = [0, 1]
    x_vals = np.linspace(x_range[0], x_range[1], 51)
    dx = x_vals[1] - x_vals[0]
    x_initial = initial_func(x_vals)
    t_max = 1
    t_stepsG = 10
    t_stepsF = 10
    t_vals, u_vals = solve_burgers((0,t_max), x_vals, t_stepsF*t_stepsG, x_initial, NU)
    # plot_burgers(t_vals, x_vals, u_vals)
    
    para_iterations = 10
    # t_coarse, u_coarse, t_fine, u_fine = pr.parareal(0, t_max, t_stepsG, t_stepsF, para_iterations, x_initial,
    #                                                  integ_burgers, integ_burgers, integ_args=(dx,), nu=NU, full_output=True)
    
    # plot_burgers(t_coarse, x_vals, u_coarse[:, :, 0].T, f'Coarse Burgers : Iteration 0', 'coarse_iteration0')
    # for k in range(1, para_iterations):
    #     plot_burgers_fine(t_fine, x_vals, u_fine[:, :, :, k].swapaxes(0,2), f'Fine Burgers : Iteration {k}', f'fine_iteration{k}')
    #     plot_burgers(t_coarse, x_vals, u_coarse[:, :, k].T, f'Coarse Burgers : Iteration {k}', f'coarse_iteration{k}')
    
    # errors_tmax([4, 1, 0.25, 0.17, 0.1], x_vals, x_initial, t_stepsG, t_stepsF, para_iterations)
    errors_discretisation([1, 2, 4], [1, 4, 16], dx, t_stepsF, x_range, initial_func, t_stepsG, t_max, para_iterations)
