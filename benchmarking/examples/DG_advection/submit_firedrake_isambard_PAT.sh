#!/bin/bash
#PBS -q arm
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -N FD_test

# The directory which contains your firedrake installation
my_firedrake=${HOME}/firedrake
# The script you want to run
myScript=DG_advection.py
# The number of processors to use
nprocs=16

# The following lines should not require modification #######

# Change to the directory that the job was submitted from
cd ${PBS_O_WORKDIR}

# Set the number of threads to 1
# This prevents any system libraries from automatically using threading.
export OMP_NUM_THREADS=1

module swap PrgEnv-cray PrgEnv-gnu
module load cray-python
module load cray-hdf5-parallel
module load perftools-preload

# Set compiler for PyOP2
export CC=cc
export CXX=CC

echo "Activating Firedrake virtual environment"
. ${my_firedrake}/firedrake/bin/activate

export MPICH_GNI_FORK_MODE=FULLCOPY

# Run Firedrake
python_with_path=`which python`
echo "Python is ${python_with_path}"
aprun -b -j 1 -n ${nprocs} pat_run ${python_with_path} ${myScript}

echo "All done"

# End of file ################################################################

