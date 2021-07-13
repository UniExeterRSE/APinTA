from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt

A = 1
B = 3

def RK4(func: Callable, X0: Iterable[float], dt: float, n: int, **func_kwargs):
    param_n = len(X0)
    x_n = np.empty((n, param_n))
    x_n[0, :] = X0
    
    for i in range(1,n):
        k1 = np.array(func(*x_n[i-1, :], **func_kwargs))
        k2 = np.array(func(*(x_n[i-1, :] + k1*dt*0.5), **func_kwargs))
        k3 = np.array(func(*(x_n[i-1, :] + k2*dt*0.5), **func_kwargs))
        k4 = np.array(func(*(x_n[i-1, :] + k3*dt), **func_kwargs))
        x_n[i, :] = x_n[i-1, :] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        
    return x_n.T

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
    time_steps = 640
    dt = 12/time_steps
    t = np.arange(0, 12, dt)
    x, y = RK4(brusselator, (0,1), dt, time_steps, A=A, B=B)
    
    draw_plots2d(x, y, t, 'Brusselator')
    
if __name__ == '__main__':
    main()