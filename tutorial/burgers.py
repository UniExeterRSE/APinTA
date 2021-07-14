from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as optimize

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

def burgers_scipy(t, u_n, dt, dx, nu=0.02, tol=1e-5, max_iterations=10, x_vals=None):
    u_nplus = optimize.fsolve(burgers_scipy_func, u_n, (u_n, dt, dx, nu))
    burgers_scipy_func(u_nplus, u_n, dt, dx, nu)
    return u_nplus

def burgers(t, u_n, dt, dx, nu=0.02, tol=1e-5, max_iterations=10, x_vals=None):
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

def solve_burgers(t_range : Tuple[float, float], x_vals, num_t, x0, nu):
    t_vals = np.linspace(*t_range, num_t, False)
    num_x = len(x_vals)
    u_vals = np.zeros((num_t, num_x))
    dt = (t_range[1] - t_range[0])/num_t
    dx = x_vals[1] - x_vals[0]
    
    u_vals[0, :] = x0
    
    for i in range(1, num_t):
        u_vals[i, :] = burgers_scipy(t_vals[i], u_vals[i-1, :], dt, dx, nu, x_vals=x_vals)
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)
    
    surface = ax.plot_surface(x_grid, t_grid, u_vals, cmap=cm.turbo, antialiased=False)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    
    fig.colorbar(surface)
    
    plt.show()
    
if __name__ == '__main__':
    x_vals = np.linspace(0, 1, 101)
    x_initial = np.sin(2*np.pi*x_vals)
    t_max = 1
    solve_burgers((0,t_max), x_vals, int(100*t_max), x_initial, NU)
