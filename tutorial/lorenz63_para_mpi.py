import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
print(f"NPROC {nproc}")
rank = comm.Get_rank()


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, integrator):
        """
        Initialise the Parareal instance with the integrator
        """
        self.solver = integrator

    def rk4_step(self, dt, x, f, **f_kwargs):
        """
        A single timestep for function f using RK4
        """
        x1 = f(x, **f_kwargs)

        x2 = f(x + x1 * dt / 2.0, **f_kwargs)
        x3 = f(x + x2 * dt / 2.0, **f_kwargs)
        x4 = f(x + x3 * dt, **f_kwargs)
        x_n = x + dt * (x1 + 2 * x2 + 2 * x3 + x4) / 6.0
        return x_n

    def integratorStep(self, integrator, deltat, y0, f, **f_kwargs):
        """
        single step of the integrator
        """
        y = integrator(deltat, y0, f, **f_kwargs)
        return y

    def integrate(self, nsteps, f, y, deltaF, **f_kwargs):
        """
        """
        y_out = np.empty((nsteps,)+(y.shape))
        y_out[0,...] = y
        for j in range(1, nsteps):  # This is for parallel running
            # y[j, ...] = self.integratorStep(
            #     self.solver, deltaF, y[j - 1, ...], f, **f_kwargs
            # )
            y_out[j, ...] = self.integratorStep(
                self.solver, deltaF, y_out[j - 1, ...], f, **f_kwargs
            )
        return y_out
        
    def parareal(self, y0, nG, nF, deltaG, deltaF, K, f, **f_kwargs):
        """
        Parareal calculation
        nG coarse grid points
        nF fine grid points
        deltaG coarse grid delta t
        deltaF fine grid delta t
        K number of parallel iterations
        f function being integrated
        returns corrected solution, corrections
        """
        y0 = np.array(y0)
        y0_extend = y0.reshape(
                (
                    1,
                    1,
                )
                + y0.shape
            )

        u = None
        if rank == 0:
            u_tilda_init = y0_extend.repeat(K, 1)
            u_tilda = np.empty(
                (
                    (
                        nG,
                        K,
                    )
                    + (y0.shape)
                )
            )
            u_tilda[0] = u_tilda_init[0]
            # Initial coarse run through
            for i in range(1, nG):
                u_tilda[i, 0, ...] = self.integratorStep(
                    self.solver, deltaG, u_tilda[i - 1, 0, ...], f, **f_kwargs
                )


            # Array to save corrected coarse propagator estimates
            u = u_tilda.copy()
        

        u_gather = np.empty(
                (
                    (
                        nG,
                        K,
                    )
                    + (y0.shape)
                )
            )
        u_hat = np.empty(
            (
                (
                    K,
                    nF,
                )
                + (y0.shape)
            )
        )

        u_hat_gather = np.empty(
                (
                    (
                        nG,
                        K,
                        nF,
                    )
                    + (y0.shape)
                )
            )

        u_local = np.zeros(
            (
                (
                    K,
                )
                + (y0.shape)
            )
        )
        
        for k in range(1, K):
            # run fine integrator in parallel for each k interation
                #comm.Recv(u_local, source=rank - 1, tag=rank)

            u = comm.bcast(u, root=0)
            if rank > 0:
                u_hat[k - 1, ...] = self.integrate(
                    nF, f, u[rank-1, k-1,...], deltaF, **f_kwargs
                )
            elif rank == 0:
                u_hat[k-1, ...] = u[0,k-1] 


            comm.Gather(u_hat, u_hat_gather, root=0)
            # print("u_hat",k, rank, u_hat[k-1])
            # print("u_hat_gather",k, rank, u_hat_gather[rank,k-1])
            if rank == 0:
                # Predict and correct
                for i in range(1,nG):
                    u_tilda[i, k, ...] = self.integratorStep(
                        self.solver, deltaG, u[i-1, k, ...], f, **f_kwargs
                    )
                    u[i, k, ...] = (
                        u_tilda[i, k, ...]
                        + u_hat_gather[i, k - 1, -1, ...]
                        - u_tilda[i, k - 1, ...]
                    )
                    #print("zero", k,rank, u[:, k, ...], u[:,k-1,...])

                    #print(f"{k}, {i}, {u[i,k,...] }u {u[i-1,k-1,...]}")
        return u, u_hat_gather


def lorenz63(xin, sigma=10, beta=8 / 3, rho=28):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       sigma, rho, beta: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x, y, z = xin
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return np.array([xdot, ydot, zdot])


def rk4_step(dt, x, f, **f_kwargs):
    """
    A single timestep for function f using RK4
    """
    x1 = f(x, **f_kwargs)

    x2 = f(x + x1 * dt / 2.0, **f_kwargs)
    x3 = f(x + x2 * dt / 2.0, **f_kwargs)
    x4 = f(x + x3 * dt, **f_kwargs)
    x_n = x + dt * (x1 + 2 * x2 + 2 * x3 + x4) / 6.0
    return x_n


def run_parareal_l63():
    a = 0
    b = 2. 
    # Number of coarse resolution timesteps
    nG = nproc
    # number of fine resolution timesteps
    nF = 1000
    # Number of parareal iterations
    K = 10
    # Model initial conditions
    y0 = [5, -5, 20]
    # coarse grid
    xG = np.linspace(a, b, nG+1)
    deltaG = (b - a) / nG
    # u_tilda shape (n_samples, n_vars)
    #xF = np.zeros((nG, int(nF / nG) + 1))
    xF = np.zeros((nG, nF+1))
    # fine grid, for each coarse grid interval
    for i in range(nG):
        left, right = xG[i], xG[i + 1]
        #xF[i, :] = np.linspace(left, right, int(nF / nG) + 1)
        xF[i, :] = np.linspace(left, right, nF+1)

    deltaF = xF[0, 1] - xF[0, 0]
    f_kwargs = {"sigma": 10, "beta": 8 / 3, "rho": 28}
    pr = Parareal(rk4_step)
    # returns corrected solution and the u_hat
    u, u_hat = pr.parareal(
        y0, nG, nF, deltaG, deltaF, K, lorenz63, **f_kwargs
    )
    # Save the data in a file
    if rank == 0:
        output_file = h5py.File("parareal_l63_mod.h5", "w")
        xG_dset = output_file.create_dataset("xG", dtype="f", data=xG)
        u_tilda_dset = output_file.create_dataset("u", dtype="f", data=u)
        corr_dset = output_file.create_dataset("u_hat", dtype="f", data=u_hat)


def plot_data(fname="parareal_l63_mod.h5"):
    dfile = h5py.File(fname, "r")
    xG = dfile["xG"][:]
    u = dfile["u"][:]
    u_hat = dfile["u_hat"][:]
    nG, K, nvars = u.shape

    for i in range(1,10):
        ax1, ax2, ax3 = plt.figure(figsize=(10, 8)).subplots(3, 1, sharex=True)
        # ax1.set_ylim(-20, 20)
        # ax1.set_xlim(0, 10)
        # ax2.set_ylim(-25, 25)
        # ax3.set_ylim(-1, 45)

        ax1.plot(xG[1:], u[:, i, 0], "-o", lw=1.5, label="x")
        ax2.plot(xG[1:], u[:, i, 1], "-o", lw=1.5, label="y")
        ax3.plot(xG[1:], u[:, i, 2], "-o", lw=1.5, label="z")
        ax1.plot(xG[1:], u_hat[:, i - 1, -1, 0], "-o", lw=1.5, label="x corr")
        ax2.plot(xG[1:], u_hat[:, i - 1, -1, 1], "-o", lw=1.5, label="y corr")
        ax3.plot(xG[1:], u_hat[:, i - 1, -1, 2], "-o", lw=1.5, label="z corr")
        ax1.set_ylabel("X")
        ax2.set_ylabel("Y")
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="lower right")
        plt.suptitle(f"Iteration {i:03}")
        plt.savefig(f"L63_iteration_{i:03}_mod.png")
        plt.show()


if __name__ == "__main__":
    run_parareal_l63()
    if rank == 0:
        plot_data()
