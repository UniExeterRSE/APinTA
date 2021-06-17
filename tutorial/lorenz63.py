import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def lorenz63_(t, x_y_z,sigma=10,beta=8/3,rho=28):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       sigma, rho, beta: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x,y,z = x_y_z
    xdot = sigma * (y - x)
    ydot = x*(rho - z) - y 
    zdot = x*y - beta*z
    return [xdot, ydot, zdot]
 
def lorenz63(x,y,z,sigma=10,beta=8/3,rho=28):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       sigma, rho, beta: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    xdot = sigma * (y - x)
    ydot = x*(rho - z) - y 
    zdot = x*y - beta*z
    return [xdot, ydot, zdot]
    
def solve_lorenz_rk4(func, x0, y0, z0, h, n):
    """
    Solve Lorenz system using RK4 method
    """
    xn,yn,zn = np.zeros(n), np.zeros(n), np.zeros(n)
    xn[0], yn[0], zn[0] = x0,y0,z0
    
    for i in range(1,n):
        k1, l1, m1 = func(xn[i-1],yn[i-1],zn[i-1])
        k2, l2, m2 = func(xn[i-1] + k1*h*0.5, yn[i-1] + l1*h*0.5, zn[i-1] + m1*h*0.5)
        k3, l3, m3 = func(xn[i-1] + k2*h*0.5, yn[i-1] + l2*h*0.5, zn[i-1] + m2*h*0.5)
        k4, l4, m4 = func(xn[i-1] + k3*h, yn[i-1] + l3*h, zn[i-1] + m3*h)
        xn[i] = xn[i-1] + h*(k1 + 2*k2 + 2*k3 + k4)*(1/6)
        yn[i] = yn[i-1] + h*(l1 + 2*l2 + 2*l3 + l4)*(1/6)
        zn[i] = zn[i-1] + h*(m1 + 2*m2 + 2*m3 + m4)*(1/6)

    return xn, yn, zn

## Simulate the Lorenz System
def run_lorenz_rk4(initial, T, n):
    x0,y0,z0 = initial
    dt = T/n
    x, y, z = solve_lorenz_rk4(lorenz63, x0, y0, z0, dt, n)
    t = np.arange(0,T,dt)
    return t, x, y, z


def run_lorenz_scipy(dt,T):
    xi=np.array([[20.,5.,-5.]])
    t = np.arange(0,T,dt)
    x_t = np.asarray([integrate.solve_ivp(lorenz63_,(0,T),i,t_eval=t).y for i in xi])
    #x_t = np.asarray([integrate.RK45(lorenz63,(0,T),i,t_eval=t).y for i in xi])
    print(x_t.shape)
    return t, x_t 

def run_lorenz_euler(dt,T):
    dt = 0.0001
    num_steps = int(T/dt)

    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    t = np.empty(num_steps +1)

    # Set initial values
    xs[0], ys[0], zs[0] = (20, 5., -5.)
    t[0] = 0.
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz63_(dt,[xs[i], ys[i], zs[i]])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        t[i+1] = t[i]+dt
    return t, np.array([[xs,ys,zs]])

def main_scipy():
    T = 10. 
    dt = T/180.
    t, l63 = run_lorenz_scipy(dt,T)
    t_, l63_ = run_lorenz_euler(dt,T)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(l63[0,0,:], l63[0,1,:], l63[0,2,:], lw=0.5)
    ax.plot(l63_[0,0,:], l63_[0,1,:], l63_[0,2,:], lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

    ax = plt.figure().subplots(3,1)
    ax[0].plot(t, l63[0,0,:], lw=0.5)
    ax[0].plot(t_, l63_[0,0,:], lw=0.5)
    ax[1].plot(t, l63[0,1,:], lw=0.5)
    ax[1].plot(t_, l63_[0,1,:], lw=0.5)
    ax[2].plot(t, l63[0,2,:], lw=0.5)
    ax[2].plot(t_, l63_[0,2,:], lw=0.5)
    ax[0].set_ylabel("X")
    ax[1].set_ylabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].set_xlabel("Time")
    ax[0].set_title("Lorenz Attractor")
    plt.show()

def main_rk4():
    x0,y0,z0 = 20,5,-5
    T = 10.
    n = 14400
    t, x, y, z = run_lorenz_rk4([x0,y0,z0],T,n)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x,y,z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

    ax = plt.figure().subplots(1,1)
    ax.plot(t, x, lw=0.5)
    ax.plot(t, y, lw=0.5)
    ax.plot(t, z, lw=0.5)
    ax.set_ylabel("X,Y,Z")
    ax.set_xlabel("Time")
    ax.set_title("Lorenz Attractor")
    plt.show()


    ax = plt.figure().subplots(3,1)
    ax[0].plot(t, x, lw=0.5)
    ax[1].plot(t, y, lw=0.5)
    ax[2].plot(t, z, lw=0.5)
    ax[0].set_ylabel("X")
    ax[1].set_ylabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].set_xlabel("Time")
    ax[0].set_title("Lorenz Attractor")
    plt.show()


if __name__ == "__main__":
    main_scipy()
    main_rk4()



