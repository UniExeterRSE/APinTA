import parareal as para
import lorenz96 
import matplotlib.pyplot as plt
import numpy as np
import sys

def lorenz63(xin,sigma=10,beta=8/3,rho=28):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       sigma, rho, beta: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x,y,z = xin
    xdot = sigma * (y - x)
    ydot = x*(rho - z) - y 
    zdot = x*y - beta*z
    return np.array([xdot, ydot, zdot])
 
def plot_l96(X,Y,Z):
    """
    Plot X,Y,Z variables
    """
    nvars = X.shape[-1]
    nrows, ncols = 4, nvars//2
    #fig, axs = plt.subplots(nvars,figsize=(10,8), sharex=True) 
    fig, axs = plt.subplots(3,figsize=(10,8)) 
    X_xpoints = np.arange(0,X.shape[0],1)
    Y_xpoints = np.arange(0,X.shape[0],1/Y.shape[-1])
    Z_xpoints = np.arange(0,X.shape[0],1/Y.shape[-1]/Z.shape[-1])
    print(X_xpoints, X_xpoints.shape)
    print(Y_xpoints,Y_xpoints.shape)
    print(Z_xpoints,Z_xpoints.shape)
    #for i in range(nvars):
    #    axs[i].plot(X[:,i])
    #    #axs[i].plot(Y[:,0, i])
    #    #axs[i].plot(Z[:,0, 0, i])
    #plt.suptitle('X variables')
    #plt.show()
    for i in range(nvars):
        fig, axs = plt.subplots(3,figsize=(10,8)) 
        axs[0].plot(X_xpoints, X[:,i])
        axs[1].plot(Y_xpoints, np.ravel(Y[:,i,:]))
        axs[2].plot(Z_xpoints, np.ravel(Z[:,i,:,:]))
        #axs[i].plot(Y[:,0, i])
        #axs[i].plot(Z[:,0, 0, i])
        plt.suptitle('X,Y,Z variables')
        plt.show()


def main_l96():
    """
    """
    K = 8 
    J = 10 
    I = 10 
    nlevels = 3
    h, g = 1., 1.
    b, c, e, d = 1., 1., 1., 1.
    F = 20.

    
    #L96 = Lorenz96(K=K,h=h,F=F,nlevels=1)
    #L96 = Lorenz96(K=K,J=J,h=h,g=g,b=b,c=c,e=e,d=d,F=F,nlevels=2)
    L96 = lorenz96.Lorenz96(K=K,J=J,I=I,h=h,g=g,b=b,c=c,e=e,d=d,F=F,nlevels=3)
    x,y,z = L96.X_coord, L96.Y_coord, L96.Z_coord
    npoints = 10000
    t_start, t_end = 0,10
    dt = (t_end - t_start)/npoints 
    X_out = np.zeros((npoints, K))
    Y_out = np.zeros((npoints, K, J))
    Z_out = np.zeros((npoints, K, J, I))
    for i in range(npoints):
        x_,y_,z_ = L96.rk4_step(dt,[x,y,z])
        X_out[i,:] = x_
        Y_out[i,:] = y_
        Z_out[i,:] = z_
        x,y,z = x_,y_,z_
    plot_l96(X_out, Y_out, Z_out)
    
def l63_init(y0,nG):
    """
    Create initial conditions array for l63
    y0 = [x0,y0,z0]
    """
    x0,y0,z0 = y0
    x = np.zeros((nG+1))
    y = np.zeros((nG+1))
    z = np.zeros((nG+1))
    #x[0] = x0
    #y[0] = y0
    #z[0] = z0
    yG = np.stack((x,y,z),axis=-1)#.reshape(((nG+1)*3,K))    
    return yG
        

    
def main_l63():
    a = 0
    b = 10.
    nG = 180
    nF = 14400 
    K = 20
    y0 = [5,-5,20]
    xG = np.linspace(a,b,nG+1)
    deltaG = (b-a)/nG
    # yG shape (n_samples, n_vars)
    yG = l63_init(y0,nG) 
    #print(yG.shape)
    xF = np.zeros((nG, int(nF/nG)+1))
    for i in range(nG):
        left,right = xG[i], xG[i+1]
        xF[i,:] = np.linspace(left,right,int(nF/nG)+1) 
     
    deltaF = xF[0,1] - xF[0,0]
    f_kwargs = {"sigma" : 10, "beta" : 8/3, "rho" : 28}
    pr = para.Parareal()
    yG_correct, correction = pr.parareal(y0, nG, nF, yG, deltaG, deltaF, K, lorenz63, **f_kwargs)
    #print(yG_correct)
    #print(yG_correct.shape)
    print(correction.shape)
    
    for i in range(K):
        ax1,ax2,ax3 = plt.figure(figsize=(10,8)).subplots(3,1)
        ax1.plot(xG[1:], yG_correct[1:,0,i], '-o', lw=1.5, label="x")
        ax2.plot(xG[1:], yG_correct[1:,1,i], '-o', lw=1.5, label="y")
        ax3.plot(xG[1:], yG_correct[1:,2,i], '-o', lw=1.5, label="z")
        ax1.plot(xG[1:], correction[1:,0,1,i], '-o', lw=1.5, label="x corr")
        ax2.plot(xG[1:], correction[1:,1,1,i], '-o', lw=1.5, label="y corr")
        ax3.plot(xG[1:], correction[1:,2,1,i], '-o', lw=1.5, label="z corr")
        ax1.set_ylabel("X")
        ax2.set_ylabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time")
        ax1.set_title(f"Lorenz Attractor: iteration {i}")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig(f"iteration_mod_{i}.png")
        plt.show()

    for i in range(K):
        ax1,ax2,ax3 = plt.figure(figsize=(10,8)).subplots(3,1)
        ax1.plot(xG[1:], yG_correct[1:,0,i]-correction[1:,0,1,i], '-o', lw=1.5, label="x")
        ax2.plot(xG[1:], yG_correct[1:,1,i]-correction[1:,1,1,i], '-o', lw=1.5, label="y")
        ax3.plot(xG[1:], yG_correct[1:,2,i]-correction[1:,2,1,i], '-o', lw=1.5, label="z")
        ax1.set_ylabel("X")
        ax2.set_ylabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time")
        ax1.set_title(f"Lorenz Attractor: iteration {i}")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.savefig(f"iteration_diff_mod_{i}.png")
        plt.show()


if __name__ == "__main__":
    main_l63()

