"""
Parareal algorithm implementation to solve Lorenz 63
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

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

def lorenz_rk4(func, x0, y0, z0, h):
    """
    Solve Lorenz system using RK4 method
    """
    
    k1, l1, m1 = func(x0,y0,z0)
    k2, l2, m2 = func(x0 + k1*h*0.5, y0 + l1*h*0.5, z0 + m1*h*0.5)
    k3, l3, m3 = func(x0 + k2*h*0.5, y0 + l2*h*0.5, z0 + m2*h*0.5)
    k4, l4, m4 = func(x0 + k3*h, y0 + l3*h, z0 + m3*h)
    x1 = x0 + h*(k1 + 2*k2 + 2*k3 + k4)*(1/6)
    y1 = y0 + h*(l1 + 2*l2 + 2*l3 + l4)*(1/6)
    z1 = z0 + h*(m1 + 2*m2 + 2*m3 + m4)*(1/6)

    return x1, y1, z1



def coarseEval(coarseIntegrator, deltaG, nG, x, y, z, f):
    """
    Coarse integrator evaluation of f
    """
    #return coarseIntegrator(f, x, y, z, deltaG, nG)
    return coarseIntegrator(f, x, y, z, deltaG)

def fineEval(fineIntegrator, deltaF, nF, x, y, z, f):
    """
    Fine integrator evaluation of f
    """
    #return fineIntegrator(f, x, y, z, deltaF, nF)
    return fineIntegrator(f, x, y, z, deltaF)

def parareal(a,b,nG,nF,K,y0,f,G,F):
    """
    inputs:
        a lower limit of domain
        b uppoer limit of domain
        nG number of coarse grid points
        nF number of fine grid points
        K number of parallel iterations
        y0 initial value  [x,y,z]
        f function being integrated  
        G coarse integrator
        F fine integrator
    """
    # Lorenz system has x,y, components. 
    # 1,2,3 suffix are the x,y,z dimensions
    xG = np.linspace(a,b,nG+1)
    yG = np.zeros((3,len(xG),K))
    deltaG = (b-a)/nG
    #print(deltaG, nG)
    yG[:,0,:] = np.array([i * np.ones(K) for i in y0])
    xF = np.zeros((nG, int(nF/nG)+1))

    # initial coarse integration solution
    #sys.exit()
    for i in range(1,nG+1):
        yG[0,i,0], yG[1,i,0],yG[2,i,0] = coarseEval(G, deltaG, nG, yG[0,i-1,0], yG[1,i-1,0], yG[2,i-1,0], f)
    # Correction terms
    yG_correct = yG.copy()
    # fine integrator
    for i in range(nG):
        left,right = xG[i], xG[i+1]
        xF[i,:] = np.linspace(left,right,int(nF/nG)+1) 

    corr = np.zeros((3,nG,int(nF/nG)+1,K))
    deltaF = xF[0,1] - xF[0,0]
    corr[:,0,0,:] = np.array([i * np.ones(K) for i in y0])
    for k in range(1,K):
        # run fine integration in parallel for each k iteration
        for i in range(nG):
            corr[:,i,0,k] = yG_correct[:,i,k-1]
            for j in range(1,int(nF/nG)+1): # This needs to be done in parallel
                corr[0,i,j,k], corr[1,i,j,k], corr[2,i,j,k] = fineEval(F, deltaF, nF, corr[0,i,j-1,k], corr[1,i,j-1,k], corr[2,i,j-1,k], f)
        # predict and correct
        for i in range(nG):
            yG[0,i+1,k], yG[1,i+1,k],yG[2,i+1,k] = coarseEval(G, deltaG, nG, yG_correct[0,i,k], yG_correct[1,i,k], yG_correct[2,i,k], f)
            #print(corr[:,i,-1,k])
            yG_correct[:,i+1,k] = yG[:,i+1,k] - yG[:,i+1,k-1] + corr[:,i,-1,k]

    for i in range(K):
        ax1,ax2,ax3 = plt.figure(figsize=(10,8)).subplots(3,1)
        ax1.plot(xG[1:], yG_correct[0,1:,i], '-o', lw=1.5, label="x")
        ax2.plot(xG[1:], yG_correct[1,1:,i], '-o', lw=1.5, label="y")
        ax3.plot(xG[1:], yG_correct[2,1:,i], '-o', lw=1.5, label="z")
        ax1.plot(xG[1:], corr[0,:,-1,i], '-o', lw=1.5, label="x corr")
        ax2.plot(xG[1:], corr[1,:,-1,i], '-o', lw=1.5, label="y corr")
        ax3.plot(xG[1:], corr[2,:,-1,i], '-o', lw=1.5, label="z corr")
        ax1.set_ylabel("X")
        ax2.set_ylabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time")
        ax1.set_title(f"Lorenz Attractor: iteration {i}")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig(f"iteration_{i}.png")
        plt.show()

    for i in range(K):
        ax1,ax2,ax3 = plt.figure(figsize=(10,8)).subplots(3,1)
        ax1.plot(xG[1:], yG_correct[0,1:,i]-corr[0,:,-1,i], '-o', lw=1.5, label="x")
        ax2.plot(xG[1:], yG_correct[1,1:,i]-corr[1,:,-1,i], '-o', lw=1.5, label="y")
        ax3.plot(xG[1:], yG_correct[2,1:,i]-corr[2,:,-1,i], '-o', lw=1.5, label="z")
        ax1.set_ylabel("X")
        ax2.set_ylabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time")
        ax1.set_title(f"Lorenz Attractor: iteration {i}")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig(f"iteration_diff{i}.png")
        plt.show()


def fineRes(a,b,nF,y0,f,F):
    """
    Lorenz results using fine resolution integrator as a comparison
    """
    #a = 0
    #b = 10.
    #nF = 500000 
    #y0 = [20,5,-5]
    #f = lorenz63
    #F = lorenz_rk4
    xF = np.linspace(a,b,(nF+1))
    yF = np.zeros((3,(nF+1)))
    yF[:,0] = np.array([i for i in y0])
    deltaF = xF[1] - xF[0]
    for i in range(1,nF+1):
        yF[0,i], yF[1,i],yF[2,i] = fineEval(F, deltaF, nF, yF[0,i-1], yF[1,i-1], yF[2,i-1], f)
    
    ax1,ax2,ax3 = plt.figure(figsize=(10,8)).subplots(3,1)
    ax1.plot(xF, yF[0,:], '-o', lw=1.5, label="x")
    ax2.plot(xF, yF[1,:], '-o', lw=1.5, label="y")
    ax3.plot(xF, yF[2,:], '-o', lw=1.5, label="z")
    ax1.set_ylabel("X")
    ax2.set_ylabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_xlabel("Time")
    ax1.set_title(f"Lorenz Attractor {deltaF}")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.savefig(f"fineres.png")
    plt.show()


def main():
    a = 0
    b = 10.
    nG = 180
    nF = 14400 
    K = 20
    y0 = [5,-5,20]
    f = lorenz63
    G = lorenz_rk4
    F = lorenz_rk4

    parareal(a,b,nG,nF,K,y0,f,G,F)
    #fineRes(a,b,nF,y0,f,F)

if __name__ == "__main__":
    main()
