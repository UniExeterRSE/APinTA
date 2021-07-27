from typing import Callable
import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

PLOT_ITERATION = False
INT_FUNC_TYPE = Callable[
    [np.ndarray, float, float, float, np.ndarray, np.ndarray, float, Callable],
    np.ndarray]

def _u_dudx(u, dx):
    u_iplus = np.roll(u, -1)
    u_iminus = np.roll(u, 1)
    result = u*(u_iplus - u_iminus)/(2*dx)
    result[0] = 0
    result[-1] = 0
    return result

def _nu_d2udx2(u, dx, nu):
    u_iplus = np.roll(u, -1)
    u_iminus = np.roll(u, 1)
    result = nu*(u_iplus - 2*u + u_iminus)/dx**2
    result[0] = 0
    result[-1] = 0
    return result

def burgers_explicitRK(u_n, dt, dx, nu, _=None, x_vals=None, t=None, Q_func=None):
    q_1 = 0
    q_n = 0
    if Q_func and x_vals is not None and t is not None:
        q_n = Q_func(t, x_vals, nu)
        q_1 = Q_func(t + dt/2, x_vals, nu)
    u_1 = u_n + dt*(_nu_d2udx2(u_n, dx, nu) + q_n) - dt/2*_u_dudx(u_n, dx)
    return u_n + dt*(_nu_d2udx2(u_1, dx, nu) + q_1- _u_dudx(u_1, dx))

def _imexRK_scipy_func(u, u_n_part, dt, dx, nu):
    zero = u - u_n_part - dt*_nu_d2udx2(u, dx, nu)
    return zero

def burgers_imexRK(u_n, dt, dx, nu, thing=None, x_vals=None, t=None, Q_func=None):
    q_1 = 0
    if Q_func and x_vals is not None and t is not None:
        q_1 = Q_func(t + dt/2, x_vals, nu)
    u_n_dep = u_n + dt*q_1 - dt/2*_u_dudx(u_n, dx)
    u_1 = optimize.fsolve(_imexRK_scipy_func, u_n, (u_n_dep, dt, dx, nu))
    return u_n + dt*(_nu_d2udx2(u_1, dx, nu) + q_1 - _u_dudx(u_1, dx))

def _scipy_func(u_nplus, u_n, dt, dx, nu, q_n):
    zero = u_nplus - u_n - dt*(_nu_d2udx2(u_nplus, dx, nu) + q_n - _u_dudx(u_nplus, dx))
    return zero

def burgers_scipy(u_n, dt, dx, nu, _=None, x_vals=None, t=None, Q_func=None):
    q_nplus = 0
    if Q_func and x_vals is not None and t is not None:
        q_nplus = Q_func(t+dt, x_vals, nu)
    u_nplus = optimize.fsolve(_scipy_func, u_n, (u_n, dt, dx, nu, q_nplus))
    return u_nplus

def burgers_fixed_point(u_n, dt, dx, nu, _=None, x_vals=None, t=None, Q_func=None, tol=1e-5, max_iterations=10):
    u_k = u_n
    q_nplus = 0
    if Q_func and x_vals is not None and t is not None:
        q_nplus = Q_func(t+dt, x_vals, nu)
    if PLOT_ITERATION:
        fig = plt.figure()
        fig.suptitle(f'Time: {t}')
        ax = fig.subplots(max_iterations+1, 1)
    for i in range(max_iterations):
        u_kminus = u_k
        first_derivative = _u_dudx(u_k, dx)
        second_derivative = _nu_d2udx2(u_k, dx, nu)
        if PLOT_ITERATION:
            ax[i].plot(x_vals, u_k)
            ax[i].plot(x_vals, first_derivative)
            ax[i].plot(x_vals, second_derivative)
        u_k = u_n + dt*(second_derivative + q_nplus - first_derivative)
        if max(abs(u_k-u_kminus)) < tol:
            break
    if PLOT_ITERATION:
        ax[max_iterations].plot(x_vals, u_k)
        plt.show()
    return u_k

def _get_departure_points(x_arrival, u_n: interp1d, u_nminus: interp1d, dt, tol=1e-8, max_iterations=10):
    x_dep = x_arrival
    u_n_arr = u_n(x_arrival)
    for i in range(max_iterations):
        u_mid = 1/2*(2*u_n(x_dep) - u_nminus(x_dep) + u_n_arr)
        x_dep_new = x_arrival - dt*u_mid
        err = np.max(np.abs(x_dep - x_dep_new))
        if np.all(err < tol):
            return x_dep_new
        x_dep = x_dep_new
    return x_dep_new

def burgers_SL(u_n, dt, dx, nu, u_nminus, x_vals, t=None, Q_func=None):
    u_n_interp_lin = interp1d(x_vals, u_n, 'linear', fill_value='extrapolate', assume_sorted=True)
    u_nminus_interp_lin = interp1d(x_vals, u_nminus, 'linear', fill_value='extrapolate', assume_sorted=True)
    x_dep = _get_departure_points(x_vals, u_n_interp_lin, u_nminus_interp_lin, dt)
    q_nplus = 0
    if Q_func and t:
        q_nplus = Q_func(t+dt, x_vals, nu)
    u_n_interp_cub = interp1d(x_vals, u_n, 'cubic', fill_value='extrapolate', assume_sorted=True)
    u_n_star = u_n_interp_cub(x_dep) + dt*q_nplus
    u_nplus = optimize.fsolve(_imexRK_scipy_func, u_n, (u_n_star, dt, dx, nu))
    
    return u_nplus
