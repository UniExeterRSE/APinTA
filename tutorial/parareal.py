import numpy as np

class Parareal():
    """
    Parallel-in-time algorithm
    """

    def __init__(self, integrator='rk4'):
        """
                """
        integrators = {'rk4':self._rk4_step}
        self.solver = integrators[integrator]

    def _rk4_step(self,dt, x,f, **f_kwargs):
        """
        A single timestep using RK4
        """
        x1 = f(x, **f_kwargs)  

        x2 = f(x+x1*dt/2.0, **f_kwargs)
        x3 = f(x+x2*dt/2.0, **f_kwargs)
        x4 = f(x+x3*dt/2.0, **f_kwargs)
        x_n = x + dt*(x1 + 2*x2 + 2*x3 + x4)/6.0
        return x_n
     
    def integratorStep(self,I,deltat,y0, f, **f_kwargs):
        """
        single step of the integrator
        """
        # initial coarse integration solution
        y = I(deltat, y0, f, **f_kwargs)
        return y

    def parareal(self, y0, nG, nF, yG_in, deltaG, deltaF, K, f, **f_kwargs):
        """
        Parareal calculation
        xG coarse grid points
        xF fine grid points
        deltaG coarse grid delta t
        deltaF fine grid delta t 
        K number of parallel iterations
        yG_in initial values for y at coarse resolution shape (n_samples, n_vars) 
        f function being integrated  
        """

        yG = np.zeros((yG_in.shape)+(K,))
        # yG shampe now n_samples, n_vars, K )
        yG[0,...] = np.array([i * np.ones(K) for i in y0])
        # Initial coarse run through 
        for i in range(1,nG+1):
            yG[i,...,0] = self.integratorStep(self.solver, deltaG, yG[i-1,...,0], f, **f_kwargs)

        yG_correct = yG.copy()
        #correction = np.zeros((nG,int(nF/nG)+1,K))
        correction = np.zeros((yG_in.shape)+(int(nF/nG)+1,K))
        for k in range(1,K):
            #run fine integrator in parallel for each k interation
            for i in range(nG):
                correction[i,...,0,k] = yG_correct[i,...,k-1]  
                for j in range(1,int(nF/nG)+1): # This is for parallel running
                    correction[i,...,j,k] = self.integratorStep(self.solver, deltaF, correction[i,...,j-1,k],f,**f_kwargs)  
            # Predict and correct 
            for i in range(nG):
                yG[i+1,...,k] = self.integratorStep(self.solver, deltaG, yG_correct[i,...,k],f,**f_kwargs) 
                yG_correct[i+1,...,k] = yG[i+1,...,k] - yG[i+1,...,k-1] + correction[i,...,-1,k]

        return yG_correct, correction

