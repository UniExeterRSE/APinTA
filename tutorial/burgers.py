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
    
def plot_burgers(t, x, u, title=None, save_name=None):
    shape0, shape1 = u.shape
    plot_burgers_fine(t.reshape(1, shape0), x, u.reshape(shape0, 1, shape1), title, save_name)
    
def plot_burgers_fine(t_fine, x, u_fine, title=None, save_name=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    for i in range(u_fine.shape[1]):
        x_grid, t_grid = np.meshgrid(x, t_fine[i, :])
        surface = ax.plot_surface(x_grid, t_grid, u_fine[:, i, :], cmap=cm.turbo, antialiased=False)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    if title:
        ax.set_title(title)
    
    fig.colorbar(surface)
    if save_name:
        plt.savefig(f"tutorial/figs/burgers/{save_name}.png")
    
    plt.show()
    
if __name__ == '__main__':
    x_vals = np.linspace(0, 1, 51)
    dx = x_vals[1] - x_vals[0]
    x_initial = np.sin(2*np.pi*x_vals)
    t_max = 1
    t_stepsG = int(t_max*10)
    t_stepsF = int(t_max*100)
    # t_vals, u_vals = solve_burgers((0,t_max), x_vals, t_stepsF, x_initial, NU)
    # plot_burgers(t_vals, x_vals, u_vals)
    
    para_iterations = 10
    t_coarse, u_coarse, t_fine, u_fine = pr.parareal(0, t_max, t_stepsG, t_stepsF, para_iterations, x_initial,
                                                     integ_burgers, integ_burgers, integ_args=(dx,), nu=NU, full_output=True)
    
    plot_burgers(t_coarse, x_vals, u_coarse[:, :, 0].T, f'Coarse Burgers : Iteration 0', 'coarse_iteration0')
    for k in range(1, para_iterations):
        plot_burgers_fine(t_fine, x_vals, u_fine[:, :, :, k].swapaxes(0,2), f'Fine Burgers : Iteration {k}', f'fine_iteration{k}')
        plot_burgers(t_coarse, x_vals, u_coarse[:, :, k].T, f'Coarse Burgers : Iteration {k}', f'coarse_iteration{k}')
