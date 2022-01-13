import numpy as np

class Parareal():
    """
    Parallel-in-time algorithm
    """

    def __init__(self, integrator):
        """
        """
        self.solver = integrator

   # def _rk4_step(self,dt, x,f, **f_kwargs):
   #     """
   #     A single timestep using RK4
   #     """
   #     x1 = f(x, **f_kwargs)  

   #     x2 = f(x+x1*dt/2.0, **f_kwargs)
   #     x3 = f(x+x2*dt/2.0, **f_kwargs)
   #     x4 = f(x+x3*dt, **f_kwargs)
   #     x_n = x + dt*(x1 + 2*x2 + 2*x3 + x4)/6.0
   #     return x_n
     
    def integratorStep(self,I,deltat,y0, f, **f_kwargs):
        """
        single step of the integrator
        """
        # initial coarse integration solution
        y = I(deltat, y0, f, **f_kwargs)
        return y

    def parareal(self, y0, nG, nF, deltaG, deltaF, K, f, **f_kwargs):
        """
        Parareal calculation
        nG coarse grid points
        nF fine grid points
        deltaG coarse grid delta t
        deltaF fine grid delta t 
        K number of parallel iterations
        f function being integrated  
        """
        y0 = np.array(y0)
        print(y0.shape)
        y0_extend = y0.reshape((1,1,)+y0.shape)
        yG_init = y0_extend.repeat(K,1)
        yG = np.empty(((nG+1,K,)+(y0.shape))) 
        yG[0] = yG_init[0]
        # Initial coarse run through 
        for i in range(1,nG+1):
            yG[i,0,...] = self.integratorStep(self.solver, deltaG, yG[i-1,0,...], f, **f_kwargs)
        # print(yG)
        yG_correct = yG.copy()
        correction = np.empty(((nG,K,int(nF/nG)+1,)+(y0.shape)))
        correction[0,:,0,...] = yG_init[0,:] 

        for k in range(1,K):
            #run fine integrator in parallel for each k interation
            for i in range(nG):
                correction[i,k,0,...] = yG_correct[i,k-1,...]  
                for j in range(1,int(nF/nG)+1): # This is for parallel running
                    correction[i,k,j,...] = self.integratorStep(self.solver, deltaF, correction[i,k,j-1,...],f,**f_kwargs)  
            # Predict and correct 
            for i in range(nG):
                yG[i+1,k,...] = self.integratorStep(self.solver, deltaG, yG_correct[i,k,...],f,**f_kwargs) 
                yG_correct[i+1,k,...] = yG[i+1,k,...] - yG[i+1,k-1,...] + correction[i,k,-1,...]

        return yG_correct, correction

