#!/bin/bash
# Number of nodes required
#SBATCH --nodes=1
# Number of tasks per node (number of MPI processes)
#SBATCH --ntasks-per-node=16
# Number of CPUs per task (number of threads set to 1)
#SBATCH --cpus-per-task=1
# Maximum time the job will run for (HH:MM:SS)
#SBATCH --time=00:10:00
# Name of the job
#SBATCH --job-name=Firedrake_GCC
# Queue for the job to run in: pq is the parallel queue
#SBATCH -p pq
# Edit this line to specify a valid project ID
#SBATCH -A Research_Project-xxxxxx
# Uncomment this line ...
# #SBATCH --mail-type=BEGIN,END,FAIL
# ... and put your email here to get notifications
# about jobs beginning, aborting or ending
# #SBATCH --mail-user=my.email@my.uni.ac.uk


# Put the name of your python script here.
# If you want to test the DG advection case the script can be downloaded from
# https://firedrakeproject.org/demos/DG_advection.py
script=DG_advection.py
# Put the location of your firedrake installation here (the directory where install_firedrake_isca_GCC.sh ran)
firedrake_dir=${HOME}/firedrake_builds/GCC

# The following lines should not require modification. 
# The number of processors is picked up by mpirun from SLURM

# Load the modules used for GCC builds
module purge
module load Python/3.6.6-foss-2018b
module load CMake/3.13.3-GCCcore-7.3.0
module load Bison/3.0.5-GCCcore-7.3.0
module load flex/2.6.4-GCCcore-7.3.0
module load libxml2/2.9.8-GCCcore-7.3.0

# Start firedrake venv
. ${firedrake_dir}/firedrake/bin/activate

# Suppress warnings from OpenMPI
export OMPI_MCA_mpi_warn_on_fork=0

# Make sure libraries do not use threading
export OMP_NUM_THREADS=1

# Run the script
echo "Running Firedrake on ${SLURM_NTASKS} processors and ${SLURM_NNODES} nodes"
mpirun -x LD_LIBRARY_PATH -x VIRTUAL_ENV -x PATH python ${script}
echo "All done"
