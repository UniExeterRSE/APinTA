from typing import Callable, Sequence, List, Tuple
import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import parareal as pr
from pr_animation import PRanimation2D
from adaptive_parareal import BaseParareal

A = 1
B = 3

def RK4(func: Callable, dt: float, n: int, x0: Sequence[float], **func_kwargs):
    param_n = len(x0)
    x_n = np.empty((n, param_n))
    x_n[0] = x0
    
    for i in range(1,n):
        k1 = np.array(func(*x_n[i-1], **func_kwargs))
        k2 = np.array(func(*(x_n[i-1] + k1*dt*0.5), **func_kwargs))
        k3 = np.array(func(*(x_n[i-1] + k2*dt*0.5), **func_kwargs))
        k4 = np.array(func(*(x_n[i-1] + k3*dt), **func_kwargs))
        x_n[i] = x_n[i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        
    return x_n

def brusselator(x, y, A=1, B=3):
    y_dot = B*x - x**2*y
    # x_dot = A + x**2*y - (B+1)*x
    x_dot = A - x - y_dot
    return (x_dot, y_dot)

def scipy_brusselator(t, y):
    return np.array(brusselator(*y, A=A, B=B))

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
    time_steps_fine = 20
    time_steps_gross = 32
    time_range = [0, 12]
    initial_cond = (0,1)
    dt = time_range[1]/(time_steps_fine*time_steps_gross)
    t = np.arange(time_range[0], time_range[1], dt)
    x, y = RK4(brusselator, dt, time_steps_fine*time_steps_gross, initial_cond, A=A, B=B).T
    
    draw_plots2d(x, y, t, 'Brusselator')
    
    t_gross, x_gross_corr, t_fine, x_fine_corr = pr.parareal(time_range[0], time_range[1], time_steps_gross, time_steps_fine, 6, initial_cond, RK4, RK4, brusselator, full_output=True, A=A, B=B)
    pr.plot_fine_comp(t_gross, x_gross_corr, t_fine, x_fine_corr, ['x', 'y'], 'Brusselator')
    pr.plot_2d_phase(x_gross_corr, ['x', 'y'], 'Brusselator', (x, y))
    animator = PRanimation2D(x_gross_corr, x_fine_corr, [[0,4], [0.5, 5]], ['x', 'y'], 10, title='Brusselator', line_colour=cm.get_cmap('YlOrRd_r'))
    animator.animate('tutorial/figs/brusselator.gif', 10)

class AdaptiveBrusselator(BaseParareal):
    def __init__(self, a: float, b: float, coarse_step: float, iterations: int, x_initial: np.ndarray,
                 target_accuracy: float, classical_K: int):
        self.coarse_step_size = coarse_step
        n_coarse = int((b-a)/coarse_step)
        super().__init__(a, b, n_coarse, iterations, x_initial)
        self.target_eta = target_accuracy
        self.K = classical_K
    
    def get_fine_accuracy(self, k_iteration):
        if k_iteration < self.K:
            k_pow = (k_iteration+1)/self.K
            return self.coarse_step_size**(1-k_pow) * (self.target_eta/2)**k_pow
        else:
            return self.target_eta/2
        
    def coarse_integration_func(self, a: float, b: float, x_in: np.ndarray) -> np.ndarray:
        sol = integ.solve_ivp(scipy_brusselator, (a,b), x_in)
        return sol.y[:, -1]
    
    def fine_integration(self, t_start: float, t_end: float, x_initial: np.ndarray,
                         coarse_step: int, iteration: int) -> Tuple[List[float], List[np.ndarray]]:
        accuracy = self.get_fine_accuracy(iteration)
        sol = integ.solve_ivp(scipy_brusselator, (t_start, t_end), x_initial, 'Radau', atol=accuracy, rtol=0)
        self.print(f'Iteration: {iteration}, step: {coarse_step}, evals:{sol.nfev}, accuracy:{accuracy}')
        return (list(sol.t), list(sol.y.T))
    
def adaptive_main():
    x0 = np.array([0,1])
    eta = 1e-6
    solve = AdaptiveBrusselator(0, 900, 0.1, 20, x0, eta, 20)
    solve.solve(5)
    
if __name__ == '__main__':
    # main()
    adaptive_main()