from firedrake import *
n = 30
mesh = UnitSquareMesh(n, n)

# We choose degree 2 continuous Lagrange polynomials. We also need a
# piecewise linear space for output purposes::

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

# We also need solution functions for the current and the next
# timestep. Note that, since this is a nonlinear problem, we don't
# define trial functions::

u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")
u_next = Function(V)
v = TestFunction(V)

# For this problem we need an initial condition::

x = SpatialCoordinate(mesh)
ic = project(as_vector([sin(pi*x[0]), 0]), V)

# We start with current value of u set to the initial condition, but we
# also use the initial condition as our starting guess for the next
# value of u::

u_.assign(ic)
u.assign(ic)

timestep = 1.0/1000
nu = 0.0001

# # lhs
du_trial = TrialFunction(V)
a = inner(du_trial, v)*dx
# a = inner((u - u_)/timestep, v)*dx 
# rhs
L1 = (inner(u_,v) - timestep*(inner(dot(u_,nabla_grad(u_)), v) + nu*inner(grad(u_), grad(v))))*dx
prob = LinearVariationalProblem(a,L1,u_next)
solv = LinearVariationalSolver(prob)
# :math:`\nu` is set to a (fairly arbitrary) small constant value::

outfile = File("forward_burgers.pvd")

outfile.write(project(u, V_out, name="Velocity"))

t = 0.0
end = 0.5

while (t <= end):
    
    # solve(F == 0, u)
    # u_.assign(u)
    solv.solve()
    u.assign(u_next)
    u_.assign(u)
    t += timestep
    outfile.write(project(u, V_out, name="Velocity"))
  
# A python script version of this demo can be found `here <burgers.py>`__.
