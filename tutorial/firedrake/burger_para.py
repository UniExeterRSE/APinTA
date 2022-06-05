from firedrake import *
import numpy as np


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, solv, u, u0, uk, k, k1, k2, k3, k4, nG, nF, deltaG, deltaF, K):
        """
        """
        
        self.solv = solv
        self.u = u
        self.u0 = u0
        self.uk = uk
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.nG = nG
        self.nF = nF
        self.deltaG = deltaG
        self.deltaF = deltaF
        self.K = K


    def rk4_step(self, dt, u0):

        self.uk.assign(u0)
        self.solv.solve()
        self.k1.assign(self.k)

        self.uk.assign(u0 + dt * self.k1 / 2.0)
        self.solv.solve()
        self.k2.assign(self.k)

        self.uk.assign(u0 + dt * self.k2 / 2.0)
        self.solv.solve()
        self.k3.assign(self.k)

        self.uk.assign(u0 + dt * self.k3)
        self.solv.solve()
        self.k4.assign(self.k)

        self.u.assign(u0 + dt * 1 / 6.0 * (self.k1 + 2.0 * self.k2 + 2.0 * self.k3 + self.k4))

        return self.u


    def coarseIntegratorStep(self, u0):
        """
        single step of the integrator
        """
        # initial coarse integration solution
        return self.rk4_step(self.deltaG, u0) 

    def fineIntegratorStep(self,u0):
        """
        single step of the integrator
        """
        return self.rk4_step(self.deltaF,u0) 

    def parareal(self, V_out):
        """
        Parareal calculation
        nG coarse grid points
        nF fine grid points
        deltaG coarse grid delta t
        deltaF fine grid delta t
        K number of parallel iterations
        f function being integrated
        """

        # yG = np.empty((self.nG + 1, self.K)).tolist()
        yG = [[self.u0.copy(deepcopy=True) for i in range(self.nG +1)] for k in range(self.K)]
        # Initial coarse run through
        print(f"First pass coarse integrator")
        outfile_0 = File(f"output/burgers_parareal_K0.pvd")
        for i in range(1, self.nG + 1):
            yG[0][i].assign(self.coarseIntegratorStep(yG[0][i-1]))
            outfile_0.write(project(yG[0][i], V_out, name="Velocity"))

        yG_correct = yG.copy()
        correction = [[[
            self.u0.copy(deepcopy=True) for j in range(int(self.nF/self.nG)+1)]
            for i in range(self.nG + 1)]
            for k  in range(self.K)]

        
        for k in range(1, self.K):

            outfileFine = File(f"output/burgers_parareal_K{k}_fine.pvd")
            print(f"Iteration {k} fine integrator")
            # run fine integrator in parallel for each k interation
            for i in range(self.nG+1):
                # correction[k][i][0].assign(yG_correct[k-1][i])
                correction[k-1][i][0].assign(yG_correct[k-1][i])
                for j in range(1, int(self.nF / self.nG) + 1):  # This is for parallel running
                    correction[k-1][i][j].assign(self.fineIntegratorStep(correction[k-1][i][j-1]))
                    outfileFine.write(project(correction[k-1][i][j], V_out, name="Velocity"))


            # Predict and correct
            print(f"Iteration {k} correction")
            outfile = File(f"output/burgers_parareal_K{k}.pvd")
            # for i in range(self.nG):
            for i in range(1, self.nG+1):
                yG[k][i].assign(self.coarseIntegratorStep(yG_correct[k][i-1]))
                yG_correct[k][i].assign((yG[k][i] - yG[k-1][i] + correction[k-1][i][-1]))
                print(yG[k][i].dat.data.max(), yG[k-1][i].dat.data.max(), correction[k-1][i][-1].dat.data.max())
                outfile.write(project(yG_correct[k][i], V_out, name="Velocity"))
                

        return yG_correct, correction



class Solver():
    """
    """
    def __init__(self, solv, u, u0, uk, k, k1, k2, k3, k4):
        self.solv = solv
        self.u = u
        self.u0 = u0
        self.uk = uk
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

    def rk4_step(self, dt):

        self.uk.assign(self.u0)
        self.solv.solve()
        self.k1.assign(self.k)

        self.uk.assign(self.u0 + dt * self.k1 / 2.0)
        self.solv.solve()
        self.k2.assign(self.k)

        self.uk.assign(self.u0 + dt * self.k2 / 2.0)
        self.solv.solve()
        self.k3.assign(self.k)

        self.uk.assign(self.u0 + dt * self.k3)
        self.solv.solve()
        self.k4.assign(self.k)

        self.u.assign(self.u0 + dt * 1 / 6.0 * (self.k1 + 2.0 * self.k2 + 2.0 * self.k3 + self.k4))

        return self.u


def main_parareal():
    n = 30
    mesh = UnitSquareMesh(n, n)

    # We choose degree 2 continuous Lagrange polynomials. We also need a
    # piecewise linear space for output purposes::

    V = VectorFunctionSpace(mesh, "CG", 2)
    V_out = VectorFunctionSpace(mesh, "CG", 1)


    # We also need solution functions for the current and the next
    # timestep. Note that, since this is a nonlinear problem, we don't
    # define trial functions::

    u0 = Function(V, name="Velocity")
    uk = Function(V, name="Velocity")
    k = Function(V, name="Velocity")
    k1 = Function(V, name="Velocity")
    k2 = Function(V, name="Velocity")
    k3 = Function(V, name="Velocity")
    k4 = Function(V, name="Velocity")

    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    # For this problem we need an initial condition::

    x = SpatialCoordinate(mesh)
    ic = project(as_vector([sin(pi * x[0]), 0]), V)

    # We start with current value of u set to the initial condition, but we
    # also use the initial condition as our starting guess for the next
    # value of u::

    u0.assign(ic)
    u.assign(ic)

    nu = 0.000001

    # # lhs
    du_trial = TrialFunction(V)
    a = inner(du_trial, v) * dx
    # rhs

    L1 = (-inner(dot(uk, nabla_grad(uk)), v) + nu * inner(grad(uk), grad(v))) * dx

    prob1 = LinearVariationalProblem(a, L1, k)
    solv1 = LinearVariationalSolver(prob1)

    # :math:`\nu` is set to a (fairly arbitrary) small constant value::

    t = 0.0
    end = 0.2

    K = 5 
    nG = 100
    nF = 200
    xG = np.linspace(t,end,nG+1)
    deltaG = (end - t)/nG
    xF = np.zeros((nG,int(nF/nG)+1))
    for i in range(nG):
        left,right = xG[i], xG[i+1]
        xF[i,:] = np.linspace(left,right,int(nF/nG)+1)
    deltaF = float(xF[0,1] - xF[0,0])
    
    solver = Parareal(solv1, u, u0, uk, k, k1, k2, k3, k4, nG, nF, deltaG, deltaF, K)
    #yG_correct shape: [K][nG]
    #correction shape: [K][nG][nF/nG+1]
    
    yG_correct, correction = solver.parareal(V_out)

    
if __name__ == "__main__":
    main_parareal()
