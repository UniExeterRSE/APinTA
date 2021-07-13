from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

NU = 0.2
PLOT_ITERATION = True

def burgers(t, u_n, dt, dx, nu=0.02, tol=1e-5, max_iterations=10, x_vals=None):
    print('\nt:', t)
    print(u_n)
    u_k = u_n
    if PLOT_ITERATION:
        fig = plt.figure()
        fig.suptitle(f'Time: {t}')
        ax = fig.subplots(max_iterations+1, 1)
    for i in range(max_iterations):
        if PLOT_ITERATION: ax[i].plot(x_vals, u_k)
        # u_k_iplus = np.pad(u_k, (0,1), 'reflect', reflect_type='odd')[1:]
        # u_k_iminus = np.pad(u_k, (1,0), 'reflect', reflect_type='odd')[:-1]
        u_k_iplus = np.roll(u_k, -1)
        u_k_iminus = np.roll(u_k, 1)
        if PLOT_ITERATION:
            ax[i].plot(x_vals, u_k_iplus - u_k_iminus)
            ax[i].plot(x_vals, u_k_iplus - 2*u_k + u_k_iminus)
        u_kminus = u_k
        u_k = u_n - dt*u_k*(u_k_iplus - u_k_iminus)/(2*dx) + dt*nu*(u_k_iplus - 2*u_k + u_k_iminus)/dx**2
        u_k[0] = 0
        u_k[-1] = 0
        # print(u_k-u_kminus)
        print(max(abs(u_k-u_kminus)))
        if max(abs(u_k-u_kminus)) < tol:
            break
    if PLOT_ITERATION:
        ax[max_iterations].plot(x_vals, u_k)
        plt.show()
    return u_k

def burgers2(t, u_n, dt, dx, nu=0.02, tol=1e-5, iterations=10, x_vals=None):
    print('\nt:', t)
    print(u_n)
    u_k = u_n
    if PLOT_ITERATION:
        fig = plt.figure()
        fig.suptitle(f'Time: {t}')
        ax = fig.subplots(iterations+1, 1)
    num_vals = len(u_n)
    for i in range(iterations):
        if PLOT_ITERATION: ax[i].plot(x_vals, u_k)
        u_kplus = np.empty_like(u_k)
        for j in range(num_vals):
            if j == 0 or j == num_vals - 1:
                u_kplus[j] = 0
                continue
            u_kplus[j] = u_n[j] - dt*u_k[j]*(u_k[j+1] - u_k[j-1])/(2*dx) + dt*nu*(u_k[j+1] - 2*u_k[j] + u_k[j-1])/dx**2
        print(max(abs(u_kplus-u_k)))
        if max(abs(u_kplus-u_k)) < tol:
            break
        u_k = u_kplus
    if PLOT_ITERATION:
        ax[iterations].plot(x_vals, u_k)
        plt.show()
    return u_k
    

def solve_burgers(t_range : Tuple[float, float],
                  x_vals, num_t, x0, nu):
    t_vals = np.linspace(*t_range, num_t, False)
    num_x = len(x_vals)
    u_vals = np.zeros((num_t, num_x))
    dt = (t_range[1] - t_range[0])/num_t
    dx = x_vals[1] - x_vals[0]
    
    u_vals[0, :] = x0
    
    for i in range(1, num_t):
        u_vals[i, :] = burgers(t_vals[i], u_vals[i-1, :], dt, dx, nu, x_vals=x_vals)
    
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
    t_max = 0.2
    solve_burgers((0,t_max), x_vals, int(100*t_max), x_initial, NU)
