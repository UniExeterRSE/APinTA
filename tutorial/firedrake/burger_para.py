from firedrake import *
import numpy as np


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, coarse_solver, fine_solver, u0,
                 nG, nF, K, save_all_output=False):
        """
        """
        
        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver        
        self.u0 = u0
        self.nG = nG
        self.nF = nF
        self.K = K
        self.save_all_output = save_all_output

    def coarseIntegratorStep(self, u0):
        """
        single step of the integrator
        """
        # initial coarse integration solution
        return self.coarse_solver.apply(u0) 

    def fineIntegratorStep(self, u0):
        """
        single step of the integrator
        """
        return self.fine_solver.apply(u0) 

    def parareal(self, V_out):
        """
        Parareal calculation
        nG coarse grid points
        nF fine grid points
        K number of parallel iterations
        """

        yG = [[self.u0.copy(deepcopy=True) for i in range(self.nG+1)] for k in range(self.K)]
        # Initial coarse run through
        print(f"First pass coarse integrator")
        outfile_0 = File(f"output/burgers_parareal_K0.pvd")
        for i in range(self.nG):
            yG[0][i+1].assign(self.coarseIntegratorStep(yG[0][i]))
            outfile_0.write(project(yG[0][i+1], V_out, name="Velocity"))

        yG_correct = [[self.u0.copy(deepcopy=True) for i in range(self.nG+1)] for k in range(self.K)]
        correction = [[[
            self.u0.copy(deepcopy=True) for j in range(int(self.nF/self.nG)+1)]
            for i in range(self.nG + 1)]
            for k  in range(self.K)]

        
        for k in range(1, self.K):

            if self.save_all_output:
                outfileFine = File(f"output/burgers_parareal_K{k}_fine.pvd")
            print(f"Iteration {k} fine integrator")
            # run fine integrator in parallel for each k interation
            for i in range(self.nG+1):
                # correction[k][i][0].assign(yG_correct[k-1][i])
                correction[k-1][i][0].assign(yG_correct[k-1][i])
                for j in range(1, int(self.nF / self.nG) + 1):  # This is for parallel running
                    correction[k-1][i][j].assign(self.fineIntegratorStep(correction[k-1][i][j-1]))
                    if self.save_all_output:
                        outfileFine.write(project(correction[k-1][i][j],
                                                  V_out, name="Velocity"))


            # Predict and correct
            print(f"Iteration {k} correction")
            outfile = File(f"output/burgers_parareal_K{k}.pvd")
            # for i in range(self.nG):
            for i in range(1, self.nG+1):
                yG[k][i].assign(self.coarseIntegratorStep(yG_correct[k][i-1]))
                yG_correct[k][i].assign((yG[k][i] - yG[k-1][i] + correction[k-1][i][-1]))
                outfile.write(project(yG_correct[k][i], V_out, name="Velocity"))
                

        return yG_correct, correction



class BurgersBE(object):
    """
    Solves Burgers equation using backwards Euler
    """
    def __init__(self, V, nu, dt):

        v = TestFunction(V)
        self.u = Function(V)
        self.u_ = Function(V)

        eqn = (self.u - self.u_) * v * dx + dt * (self.u * self.u.dx(0) *  v * dx + nu * self.u.dx(0) * v.dx(0) * dx)

        prob = NonlinearVariationalProblem(eqn, self.u)
        self.solver = NonlinearVariationalSolver(prob)

    def apply(self, u):

        self.u_.assign(u)
        self.solver.solve()

        return self.u


def main_parareal():
    n = 1000
    mesh = PeriodicUnitIntervalMesh(n)

    # We choose degree 2 continuous Lagrange polynomials. We also need a
    # piecewise linear space for output purposes::

    V = FunctionSpace(mesh, "CG", 2)
    V_out = FunctionSpace(mesh, "CG", 1)


    # We also need solution functions for the current and the next
    # timestep. Note that, since this is a nonlinear problem, we don't
    # define trial functions::

    u0 = Function(V, name="Velocity")
    uk = Function(V, name="Velocity")

    u = Function(V, name="VelocityNext")


    # We start with current value of u set to the initial condition, but we
    # also use the initial condition as our starting guess for the next
    # value of u::

    # Initial condition
    Vic = FunctionSpace(mesh, "DG", 0)
    pcg = PCG64(seed=123456789)
    rg = RandomGenerator(pcg)
    ic = rg.normal(Vic)

    du = TrialFunction(V)
    v = TestFunction(V)
    u0 = Function(V)
    u = Function(V)
    # lengthscale over which to smooth
    alpha = Constant(0.05)
    area = assemble(1*dx(domain=ic.ufl_domain()))
    a = (alpha**2 * du.dx(0) * v.dx(0) + du * v) * dx
    L = (ic / sqrt(area)) * v * dx
    solve(a == L, u0,
          solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    a = 10
    u0.interpolate(Constant(1/a)*ln(1 + exp(Constant(a)*u0)))

    # viscosity
    nu = 0.0001

    # end time
    tmax = 5

    # number of parareal iterations
    K = 10
    # number of coarse timesteps
    nG = 50
    # number of fine timesteps per coarse timestep
    nF = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print(dT, dt)

    G = BurgersBE(V, nu, dT)
    F = BurgersBE(V, nu, dt)
    solver = Parareal(G, F, u0, nG, nF, K)
    
    yG_correct, correction = solver.parareal(V_out)

    
if __name__ == "__main__":
    main_parareal()
