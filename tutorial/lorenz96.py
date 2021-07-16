import numpy as np

class Lorenz96(object):
    def __init__(self, K=1,J=1,I=1,nlevels=1,h=None,g=None,
            F=None,c=None,b=None,e=None,d=None,
            X0=None,Y0=None,Z0=None):
        """
        Follows 3 level Lorenz 96 formulation in Thornes et al (2016)
        https://doi.org/10.1002/qj.2974
        K : number of X variables
        J : number of Y variables
        I : number of Z variables
        h : coupling constant between X and Y
        g : Z damping parameter
        F : Forcing term
        c : time scale ratio between X and Y
        b : spatial scale ratio between X and Y
        e : time scale ratio between Y and Z
        d : spatial scale ratio between Y and Z
        """
        self.nlevels = nlevels
        if self.nlevels == 1:
            assert(K is not None), 'Need to set K value'
            assert(F is not None), 'Need to set F value'
         
        if self.nlevels == 2:
            assert(K is not None), 'Need to set K value'
            assert(F is not None), 'Need to set F value'
            assert(J is not None), 'Need to set J value'
            assert(h is not None), 'Need to set h value'
            assert(F is not None), 'Need to set F value'
            assert(c is not None), 'Need to set c value'
            assert(b is not None), 'Need to set b value'

        if self.nlevels == 3:
            assert(K is not None), 'Need to set K value'
            assert(F is not None), 'Need to set F value'
            assert(J is not None), 'Need to set J value'
            assert(h is not None), 'Need to set h value'
            assert(F is not None), 'Need to set F value'
            assert(c is not None), 'Need to set c value'
            assert(b is not None), 'Need to set b value'
            assert(K is not None), 'Need to set K value'
            assert(g is not None), 'Need to set g value'
            assert(e is not None), 'Need to set e value'
            assert(d is not None), 'Need to set d value'


        if X0 is not None:
             self._X = X0.copy()
        else:
             self._X = np.random.normal(loc=0,scale = 1, size=K)
        if Y0 is not None:
             self._Y = X0.copy()
        else:
             self._Y = np.random.normal(loc=0,scale = 1, size=(K,J))
        if Z0 is not None:
             self._Z = Z0.copy()
        else:
             self._Z = np.random.normal(loc=0,scale = 0.05, size=(K,J,I))
              
        self.h = h
        self.g = g
        self.F = F
        self.c = c
        self.b = b
        self.e = e
        self.d = d

    @property
    def X_coord(self):
        """
        get X coord array
        """
        return self._X
    
    @property
    def Y_coord(self):
        """
        get Y coord array
        """
        return self._Y
    
    @property
    def Z_coord(self):
        """
        get Z coord array
        """
        return self._Z
    
        
    def rk4_step(self,x,y,z,dt):
        """
        A single timestep using RK4
        """
        x1,y1,z1 = self.l96(x,y,z)  

        x2,y2,z2 = self.l96(x+x1*dt/2.0,
                y + y1*dt/2.0,
                z + z1*dt/2.0)
        x3,y3,z3 = self.l96(x+x2*dt/2.0,
                y + y2*dt/2.0,
                z + z2*dt/2.0)
        x4,y4,z4 = self.l96(x+x3*dt/2.0,
                y + y3*dt/2.0,
                z + z3*dt/2.0)
        x_n = x + dt*(x1 + 2*x2 + 2*x3 + x4)/6.0
        y_n = y + dt*(y1 + 2*y2 + 2*y3 + y4)/6.0
        z_n = z + dt*(z1 + 2*z2 + 2*z3 + z4)/6.0
        return x_n, y_n, z_n
        

    def l96(self, X,Y,Z):
       """
       The L96 model depending on the number of levels
       """
       Y_next,Z_next = 0.,0.  
       X_next = np.roll(X,1)*(np.roll(X,-1) - np.roll(X,2)) - X + self.F
       if self.nlevels > 1:
           X_next -= (self.h*self.c/self.b)*Y.sum(axis=1)
           Y_next = (self.c*self.b*np.roll(Y,-1)*np.roll(Y,-2) - np.roll(Y,1)) - self.c*Y + (self.h*self.c/self.b)*X[:,None]
       if self.nlevels > 2:
           Y_next -= (self.h*self.e/self.d)*Z.sum(axis=2)
           Z_next = self.e*self.d*np.roll(Z,1)*(np.roll(Z,-1) - np.roll(Z,2)) - self.g*self.e*Z + (self.h*self.e/self.d)*Y[:,:,None]
       return X_next, Y_next, Z_next 

def main():
    """
    """
    K = 8 
    J = 8
    I = 8
    nlevels = 3
    h, g = 1., 1.
    b, c, e, d = 10., 10., 10., 10.
    F = 20.

    
    #L96 = Lorenz96(K=K,h=h,F=F,nlevels=1)
    L96 = Lorenz96(K=K,J=J,I=I,h=h,g=g,b=b,c=c,e=e,d=d,F=F,nlevels=3)
    x,y,z = L96.X_coord, L96.Y_coord, L96.Z_coord
    dt = 0.001
    for i in range(10):
        x_,y_,z_ = L96.rk4_step(x,y,z,dt)
        x,y,z = x_,y_,z_
        print("X", x_)
        print("Y", y_)
        print("Z", z_)

if __name__ == "__main__":
    main()

 

