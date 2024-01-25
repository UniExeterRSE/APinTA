from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, coarse_solver, fine_solver, V, u0,
                 nG, K, save_all_output=False):
        """
        """

        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver        
        self.u0 = u0
        self.nG = nG
        self.K = K
        self.save_all_output = save_all_output

        self.yref = [Function(V) for i in range(self.nG+1)]
        self.yG = [Function(V) for i in range(self.nG+1)]
        self.yG_prev = [Function(V) for i in range(self.nG+1)]
        self.yG_correct = [Function(V) for i in range(self.nG+1)]
        self.yF = [Function(V) for i in range(self.nG+1)]

        self.yref[0].assign(u0)
        self.yG[0].assign(u0)
        self.yG_prev[0].assign(u0)
        self.yG_correct[0].assign(u0)
        self.yF[0].assign(u0)

        self.yG_out = Function(V, name="yG")
        self.yF_out = Function(V, name="yF")
        self.yref_out = Function(V, name="yref")

    def parareal(self):
        """
        Parareal calculation
        nG number of coarse timesteps
        K number of parallel iterations
        """

        # reference solution
        yref = self.yref
        for i in range(self.nG):
            yref[i+1].assign(self.fine_solver.apply(yref[i]))

        yG = self.yG
        yG_prev = self.yG_prev
        yG_correct = self.yG_correct
        yF = self.yF

        # Initial coarse run through
        print(f"First pass coarse integrator")
        outfile_0 = File(f"output/burgers_parareal_K0.pvd")
        self.yG_out.assign(yG[0])
        self.yref_out.assign(yref[0])
        outfile_0.write(self.yG_out, self.yref_out)

        for i in range(self.nG):
            yG[i+1].assign(self.coarse_solver.apply(yG[i]))
            yG_correct[i+1].assign(yG[i+1])
            self.yG_out.assign(yG[i+1])
            self.yref_out.assign(yref[i+1])
            outfile_0.write(self.yG_out, self.yref_out)

        for k in range(self.K):

            if self.save_all_output:
                outfileFine = File(f"output/burgers_parareal_K{k+1}_fine.pvd")

            print(f"Iteration {k+1} fine integrator")
            # run fine integrator on each time slice
            xf = []
            yf = []
            zf = []
            for i in range(self.nG):
                yF[i].assign(yG_correct[i])
                yF[i+1].assign(self.fine_solver.apply(yF[i]))
                xf.append(self.yF[i+1].dat.data[0,0])
                yf.append(self.yF[i+1].dat.data[0,1])
                zf.append(self.yF[i+1].dat.data[0,2])

                if self.save_all_output:
                    self.yF_out.assign(yF[i+1])
                    outfileFine.write(self.yF_out)

            # Predict and correct
            print(f"Iteration {k+1} correction")
            outfile = File(f"output/burgers_parareal_K{k+1}.pvd")
            x = []
            y = []
            z = []
            for i in range(self.nG):
                yG_prev[i+1].assign(yG_correct[i])
                yG[i+1].assign(self.coarse_solver.apply(yG_correct[i]))
                yG_correct[i+1].assign(yG[i+1] - yG_prev[i+1] + yF[i+1])
                self.yG_out.assign(yG_correct[i])
                self.yref_out.assign(yref[i])
                outfile.write(self.yG_out, self.yref_out)

                x.append(self.yG_out.dat.data[0,0])
                y.append(self.yG_out.dat.data[0,1])
                z.append(self.yG_out.dat.data[0,2])

            ax1, ax2, ax3 = plt.figure(figsize=(10, 8)).subplots(3, 1)
            ax1.plot(x)
            ax2.plot(y)
            ax3.plot(z)
            ax1.plot(xf)
            ax2.plot(yf)
            ax3.plot(zf)
            plt.show()


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

        eqn = (self.u - self.u_) * v * dx + dt * (self.u_ * self.u_.dx(0) *  v * dx + nu * self.u.dx(0) * v.dx(0) * dx)

        prob = NonlinearVariationalProblem(eqn, self.u)
        self.solver = NonlinearVariationalSolver(prob)

        self.ndt = ndt

    def apply(self, u):

        for n in range(self.ndt):
            self.u_.assign(u)
            self.solver.solve()

        return self.u


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
        self.dt = dt

    def apply(self, X):

        self.X.assign(X)
        self.solver.solve()
        self.k1.assign(self.k)

        self.X.assign(X + self.dt * self.k1 / 2.0)
        self.solver.solve()
        self.k2.assign(self.k)

        self.X.assign(X + self.dt * self.k2 / 2.0)
        self.solver.solve()
        self.k3.assign(self.k)

        self.X.assign(X + self.dt * self.k3)
        self.solver.solve()
        self.k4.assign(self.k)

        self.X.assign(X + self.dt * 1 / 6.0 * (self.k1 + 2.0 * self.k2 + 2.0 * self.k3 + self.k4))

        return self.X


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
    tmax = 0.5

    # number of parareal iterations
    K = 5
    # number of coarse timesteps
    nG = 2
    # number of fine timesteps per coarse timestep
    nF = 1000

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print("coarse timestep: ", dT)
    print("fine timestep: ", dt)

    # G = BurgersBE(V, nu, dT, 1)
    # F = BurgersBE(V, nu, dt, nF)
    G = BurgersIMEX(V, nu, dT, 1)
    F = BurgersIMEX(V, nu, dt, nF)
    solver = Parareal(G, F, V, u0, nG, K, save_all_output=True)
    
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

    K = 2
    nG = 180
    nF = 14400

    tmax = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    G = RK4Lorenz63(V, dT, 1)
    F = RK4Lorenz63(V, dt, nF)
    
    solver = Parareal(G, F, V, X, nG, K)
    solver.parareal()


if __name__ == "__main__":
    # main_parareal()
    lorenz_parareal()
    
