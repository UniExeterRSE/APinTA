from firedrake import *
import numpy as np




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


def rk4_step(dt, solv, u, u0, uk, k, k1, k2, k3, k4):
    """
    RK$ step
    """
    uk.assign(u0)
    solv.solve()
    k1.assign(k)

    uk.assign(u0 + dt * k1 / 2.0)
    solv.solve()
    k2.assign(k)

    uk.assign(u0 + dt * k2 / 2.0)
    solv.solve()
    k3.assign(k)

    uk.assign(u0 + dt * k3)
    solv.solve()
    k4.assign(k)

    u.assign(u0 + dt * 1 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    return u
  
def main():
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

    timestep = 1.0 / 200
    nu = 0.0001

    # # lhs
    du_trial = TrialFunction(V)
    a = inner(du_trial, v) * dx
    # rhs

    L1 = (-inner(dot(uk, nabla_grad(uk)), v) + nu * inner(grad(uk), grad(v))) * dx

    prob1 = LinearVariationalProblem(a, L1, k)
    solv1 = LinearVariationalSolver(prob1)

    # :math:`\nu` is set to a (fairly arbitrary) small constant value::

    outfile = File("forward_rk4.pvd")

    outfile.write(project(u, V_out, name="Velocity"))

    t = 0.0
    end = 0.5

    solver = Solver(solv1, u, u0, uk, k, k1, k2, k3, k4)
    
    while t <= end:

        # u = rk4_step(timestep, solv1, u, u0, uk, k, k1, k2, k3, k4)
        u = solver.rk4_step(timestep)

        u0.assign(u)
        t += timestep
        outfile.write(project(u, V_out, name="Velocity"))


if __name__ == "__main__":
    main()
