import sympy as sp
from sympy.core import Expr, Symbol
import numpy as np
from typing import Callable, Tuple
import inspect

def _get_Q(u_sym: Expr, x_sym: Symbol, t_sym: Symbol, nu: Symbol, *args: Tuple[Symbol]) -> Callable:
    q_sym = u_sym.diff(t_sym) + u_sym*u_sym.diff(x_sym) - nu*u_sym.diff(x_sym, 2)
    return sp.lambdify([*args, t_sym, x_sym, nu], q_sym, 'numpy')

def _fix_func_str(func_str: str):
    """Removes the first line and and np. in front of numpy funcs"""
    func_str = '\n'.join(func_str.split('\n')[1:])
    func_str = func_str.replace('sin', 'np.sin')
    func_str = func_str.replace('cos', 'np.cos')
    func_str = func_str.replace('pi', 'np.pi')
    func_str = func_str.replace('builtins.sum', 'sum')
    return func_str

def get_funcs_source_code(u_sym: Expr, x_sym: Symbol, t_sym: Symbol, nu: Symbol, name: str, *args: Tuple[Symbol, str]):
    """Print out the source code for the lambdified version of a sympy
    expression and it's associated Q function.
    
    This is done because lambda functions cannot be pickled and so
    sympy.lambdify generates a function that cannot be passed into
    a multiprocessing pool.
    """
    sym_args = []
    extra_args = ''
    for sym, sym_type in args:
        sym_args.append(sym)
        extra_args += f'{sym.name}: {sym_type}, '
        
    sol_func = sp.lambdify((*sym_args, t_sym, x_sym, nu), u_sym)
    q_func = _get_Q(u_sym, x_sym, t_sym, nu, *sym_args)
        
    
    print(f'def {name}_sol({extra_args}t, x):')
    print(_fix_func_str(inspect.getsource(sol_func)))
    
    print(f'def {name}_q({extra_args}t, x, nu: float):')
    print(_fix_func_str(inspect.getsource(q_func)))
    
def B1_sol(k: int, t, x):
    return (np.sin(2*np.pi*t)*np.sin(2*np.pi*x) + np.sin(2*np.pi*k*t)*np.sin(2*np.pi*k*x)/k)

def B1_q(k: int, t, x, nu: float):
    return (4*np.pi**2*nu*(k*np.sin(2*np.pi*k*t)*np.sin(2*np.pi*k*x) + np.sin(2*np.pi*t)*np.sin(2*np.pi*x)) + (np.sin(2*np.pi*t)*np.sin(2*np.pi*x) + np.sin(2*np.pi*k*t)*np.sin(2*np.pi*k*x)/k)*(2*np.pi*np.sin(2*np.pi*t)*np.cos(2*np.pi*x) + 2*np.pi*np.sin(2*np.pi*k*t)*np.cos(2*np.pi*k*x)) + 2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*t) + 2*np.pi*np.sin(2*np.pi*k*x)*np.cos(2*np.pi*k*t))

def B2_sol(k_max: int, epsilon: float, t, x):
    return (0.5*(sum(epsilon*np.sin(-np.pi*k*t + 2*np.pi*k*x + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))))

def B2_q(k_max: int, epsilon: float, t, x, nu: float):
    return (2.0*np.pi**2*epsilon*nu*(sum(k**2*np.sin(np.pi*k*(-t + 2*x + 1))/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))) + 0.25*(sum(epsilon*np.sin(-np.pi*k*t + 2*np.pi*k*x + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1)))*(sum(2*np.pi*epsilon*k*np.cos(-np.pi*k*t + 2*np.pi*k*x + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))) + 0.5*(sum(-np.pi*epsilon*k*np.cos(-np.pi*k*t + 2*np.pi*k*x + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))))

# Benchmark 2 shifted by 0.5 in x to determine of issues are boundary related
def B2shift_sol(k_max: int, epsilon: float, t, x):
    return (0.5*(sum(epsilon*np.sin(-np.pi*k*t + 2*np.pi*k*(x - 0.5) + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))))        

def B2shift_q(k_max: int, epsilon: float, t, x, nu: float):
    return (-2.0*np.pi**2*epsilon*nu*(sum(k**2*np.sin(np.pi*k*(t - 2*x))/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))) + 0.25*(sum(epsilon*np.sin(-np.pi*k*t + 2*np.pi*k*(x - 0.5) + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1)))*(sum(2*np.pi*epsilon*k*np.cos(-np.pi*k*t + 2*np.pi*k*(x - 0.5) + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))) + 0.5*(sum(-np.pi*epsilon*k*np.cos(-np.pi*k*t + 2*np.pi*k*(x - 0.5) + np.pi*k)/np.sinh(0.5*np.pi*epsilon*k) for k in range(1, k_max+1))))

def main():
    x = sp.Symbol('x')
    t = sp.Symbol('t')
    nu = sp.Symbol('nu')
    k = sp.Symbol('k')

    u_sol_B1 = sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*t) + 1/k*sp.sin(2*k*sp.pi*x)*sp.sin(2*k*sp.pi*t)
    get_funcs_source_code(u_sol_B1, x, t, nu, 'B1', (k, 'int'))
    
    k_max = sp.Symbol('k_max')
    epsilon = sp.Symbol('epsilon')
    phi = epsilon/sp.sinh(1/2*epsilon*sp.pi*k)
    summed_func = sp.sin(2*sp.pi*k*x - sp.pi*k*t + sp.pi*k)*phi
    u_sol_B2 = 1/2*sp.Sum(summed_func, (k, 1, k_max))
    get_funcs_source_code(u_sol_B2, x, t, nu, 'B2', (k_max, 'int'), (epsilon, 'float'))

if __name__ == '__main__':
    main()
