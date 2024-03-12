from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, coarse_solver, fine_solver, V, u0, nG, K):
        """
        V: function space
        u0: initial condition
        nG: number of coarse timesteps
        K: number of parareal iterations
        """

        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver        
        self.nG = nG
        self.K = K

        # list to hold reference solution at coarse time points
        self.yref = [Function(V) for i in range(self.nG+1)]

        # list to hold coarse solution at coarse time points
        self.yG = [Function(V) for i in range(self.nG+1)]

        # list to hold coarse solution from previous iteration at
        # coarse time points
        self.yG_prev = [Function(V) for i in range(self.nG+1)]

        # list to hold solution at coarse time points
        self.soln = [Function(V) for i in range(self.nG+1)]

        # list to hold fine solution at coarse time points
        self.yF = [Function(V) for i in range(self.nG+1)]

        # Initialise everything
        self.yref[0].assign(u0)
        self.yG[0].assign(u0)
        self.yG_prev[0].assign(u0)
        self.yF[0].assign(u0)
        self.soln[0].assign(u0)

        # Functions for writing out to pvd files
        self.yG_out = Function(V, name="yG")
        self.yF_out = Function(V, name="yF")
        self.yref_out = Function(V, name="yref")

    def parareal(self):
        """
        Parareal calculation
        """

        # compute reference solution
        yref = self.yref
        for i in range(self.nG):
            # each application of the fine solver does nF timesteps
            yref[i+1].assign(self.fine_solver.apply(yref[i]))

        # get some things
        yG = self.yG
        yG_prev = self.yG_prev
        yF = self.yF
        soln = self.soln

        # set up output file and write out initial coarse solution and
        # initial reference solution (the same, as it's just the
        # initial condition!)
        outfile0 = File(f"output/burgers_parareal_K0.pvd")
        self.yG_out.assign(yG[0])
        self.yref_out.assign(yref[0])
        outfile0.write(self.yG_out, self.yref_out)

        # Initial coarse run through
        print(f"First coarse integrator iteration")
        for i in range(self.nG):
            yG[i+1].assign(self.coarse_solver.apply(yG[i]))
            soln[i+1].assign(yG[i+1])
            yG_prev[i+1].assign(yG[i+1])
            self.yG_out.assign(yG[i+1])
            self.yref_out.assign(yref[i+1])
            outfile0.write(self.yG_out, self.yref_out)

        for k in range(self.K):
            print(f"Iteration {k+1}")
            outfile = File(f"output/burgers_parareal_K{k+1}.pvd")
            self.yG_out.assign(soln[0])
            self.yref_out.assign(yref[0])
            outfile.write(self.yG_out, self.yref_out)
            # Predict and correct
            for i in range(self.nG):
                yF[i+1].assign(self.fine_solver.apply(soln[i]))
            for i in range(self.nG):
                yG[i+1].assign(self.coarse_solver.apply(soln[i]))
                soln[i+1].assign(yG[i+1] - yG_prev[i+1] + yF[i+1])
                #print(errornorm(yG[i+1], yG_prev[i+1]))
                #print(errornorm(yG_correct[i+1], yref[i+1]))
            for i in range(self.nG):
                yG_prev[i+1].assign(yG[i+1])
                self.yG_out.assign(soln[i+1])
                self.yref_out.assign(yref[i+1])
                outfile.write(self.yG_out, self.yref_out)


class BurgersBE(object):
    """
    Solves Burgers equation using backwards Euler
    """
    def __init__(self, V, nu, dt, ndt):

        v = TestFunction(V)
        self.u = Function(V)
        self.u_ = Function(V)

        eqn = (self.u - self.u_) * v * dx + dt * (self.u * self.u.dx(0) *  v * dx + nu * self.u.dx(0) * v.dx(0) * dx)

        prob = NonlinearVariationalProblem(eqn, self.u)
        self.solver = NonlinearVariationalSolver(prob)

        self.ndt = ndt

    def apply(self, u):

        for n in range(self.ndt):
            self.u_.assign(u)
            self.solver.solve()

        return self.u


class BurgersIMEX(object):
    """
    Solves Burgers equation using backwards Euler
    """
    def __init__(self, V, nu, dt, ndt):

        v = TestFunction(V)
        self.u = Function(V)
        self.u_ = Function(V)
        self.unp1 = Function(V)

        eqn = (self.u - self.u_) * v * dx + dt * (self.u_ * self.u_.dx(0) *  v * dx + nu * self.u.dx(0) * v.dx(0) * dx)

        prob = NonlinearVariationalProblem(eqn, self.u)
        self.solver = NonlinearVariationalSolver(prob)

        self.ndt = ndt

    def apply(self, u):

        self.unp1.assign(u)
        for n in range(self.ndt):
            self.u_.assign(self.unp1)
            self.solver.solve()
            self.unp1.assign(self.u)

        return self.unp1


class RK4Lorenz63(object):

    def __init__(self, V, dt, ndt, sigma=10, beta=8/3, rho=28):

        v1, v2, v3 = TestFunctions(V)
        x_, y_, z_ = TrialFunctions(V)
        self.X = Function(V)
        x, y, z = self.X.sub(0), self.X.sub(1), self.X.sub(2)
        a = v1 * x_ * dx + v2 * y_ * dx + v3 * z_ * dx
        L = (v1 * sigma * (y - x) * dx + v2 * (x * (rho - z) - y) * dx +
             v3 * (x * y - beta * z) * dx)

        self.k = Function(V)
        prob = LinearVariationalProblem(a, L, self.k)
        self.solver = LinearVariationalSolver(prob)

        self.k1 = Function(V)
        self.k2 = Function(V)
        self.k3 = Function(V)
        self.k4 = Function(V)
        self.Xn = Function(V)
        self.Xnp1 = Function(V)
        self.dt = dt
        self.ndt = ndt

    def apply(self, X):

        self.Xnp1.assign(X)

        for n in range(self.ndt):
            self.Xn.assign(self.Xnp1)
            self.solver.solve()
            self.k1.assign(self.k)

            self.X.assign(self.Xn + self.dt * self.k1 / 2.0)
            self.solver.solve()
            self.k2.assign(self.k)

            self.X.assign(self.Xn + self.dt * self.k2 / 2.0)
            self.solver.solve()
            self.k3.assign(self.k)

            self.X.assign(self.Xn + self.dt * self.k3)
            self.solver.solve()
            self.k4.assign(self.k)

            self.Xnp1.assign(self.Xn + self.dt * 1 / 6.0 * (self.k1 + 2.0 * self.k2 + 2.0 * self.k3 + self.k4))
            

        return self.Xnp1


def main_parareal():
    n = 1000
    mesh = PeriodicUnitIntervalMesh(n)

    # We choose degree 2 continuous Lagrange polynomials. We also need a
    # piecewise linear space for output purposes::

    V = FunctionSpace(mesh, "CG", 2)

    # We also need solution functions for the current and the next
    # timestep. Note that, since this is a nonlinear problem, we don't
    # define trial functions::

    u0 = Function(V, name="Velocity")

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
    nu = 0.01

    # end time
    tmax = 1

    # number of parareal iterations
    K = 0
    # number of coarse timesteps
    nG = 10
    # number of fine timesteps per coarse timestep
    nF = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print("coarse timestep: ", dT)
    print("fine timestep: ", dt)

    #G = BurgersBE(V, nu, dT, 1)
    #F = BurgersBE(V, nu, dt, nF)
    G = BurgersIMEX(V, nu, dT, 1)
    F = BurgersIMEX(V, nu, dt, nF)
    solver = Parareal(G, F, V, u0, nG, K)
    
    solver.parareal()


def gander_parareal():
    # settings to match Gander and Hairer paper
    n = 50
    mesh = PeriodicUnitIntervalMesh(n)

    # We choose degree 2 continuous Lagrange polynomials.
    V = FunctionSpace(mesh, "CG", 2)
    u0 = Function(V, name="Velocity")

    # Initial condition
    x = SpatialCoordinate(mesh)[0]
    u0.interpolate(sin(2*pi*x))

    # viscosity
    nu = 1/50.

    # end time
    tmax = 1

    # number of parareal iterations
    K = 10
    # number of coarse timesteps
    nG = 10
    # number of fine timesteps per coarse timestep
    nF = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print("coarse timestep: ", dT)
    print("fine timestep: ", dt)

    G = BurgersBE(V, nu, dT, 1)
    F = BurgersBE(V, nu, dt, nF)
    solver = Parareal(G, F, V, u0, nG, K)
    
    solver.parareal()


def lorenz_parareal():

    n = 1
    mesh = UnitIntervalMesh(n)
    V = VectorFunctionSpace(mesh, "DG", 0, dim=3)
    X = Function(V)
    x, y, z = X.sub(0), X.sub(1), X.sub(2)
    x.assign(Constant(5))
    y.assign(Constant(-5))
    z.assign(Constant(20))

    K = 20
    nG = 180
    nF = 80

    tmax = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print("coarse dt: ", dT)
    print("fine dt: ", dt)
    G = RK4Lorenz63(V, dT, 1)
    F = RK4Lorenz63(V, dt, nF)
    
    solver = Parareal(G, F, V, X, nG, K)
    solver.parareal()




if __name__ == "__main__":
    #main_parareal()
    #lorenz_parareal()
    #gander_parareal()
    get_burgers_data()
