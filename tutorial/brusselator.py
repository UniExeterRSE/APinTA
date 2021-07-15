from typing import Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
import parareal as pr

A = 1
B = 3

def RK4(func: Callable, dt: float, n: int, x0: Sequence[float], **func_kwargs):
    param_n = len(x0)
    x_n = np.empty((param_n, n))
    x_n[:, 0] = x0
    
    for i in range(1,n):
        k1 = np.array(func(*x_n[:, i-1], **func_kwargs))
        k2 = np.array(func(*(x_n[:, i-1] + k1*dt*0.5), **func_kwargs))
        k3 = np.array(func(*(x_n[:, i-1] + k2*dt*0.5), **func_kwargs))
        k4 = np.array(func(*(x_n[:, i-1] + k3*dt), **func_kwargs))
        x_n[:, i] = x_n[:, i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        
    return x_n

def brusselator(x, y, A=1, B=3):
    y_dot = B*x - x**2*y
    # x_dot = A + x**2*y - (B+1)*x
    x_dot = A - x - y_dot
    return (x_dot, y_dot)

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
    
def main():
    time_steps_fine = 640
    time_steps_gross = 32
    time_range = [0, 12]
    initial_cond = (0,1)
    dt = 12/time_steps_fine
    t = np.arange(time_range[0], time_range[1], dt)
    x, y = RK4(brusselator, dt, time_steps_fine, initial_cond, A=A, B=B)
    
    draw_plots2d(x, y, t, 'Brusselator')
    
    t_gross, x_gross_corr, t_fine, x_fine_corr = pr.parareal(time_range[0], time_range[1], time_steps_gross, time_steps_fine, 6, initial_cond, RK4, RK4, brusselator, full_output=True, A=A, B=B)
    pr.plot_fine_comp(t_gross, x_gross_corr, t_fine, x_fine_corr, ['x', 'y'], 'Brusselator')
    pr.plot_2d_phase(x_gross_corr, ['x', 'y'], 'Brusselator', (x, y))

    
if __name__ == '__main__':
    main()