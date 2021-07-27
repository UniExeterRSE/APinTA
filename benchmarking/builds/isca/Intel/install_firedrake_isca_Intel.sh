#!/bin/bash

# Script for installing Firedrake on Isca (University of Exeter HPC system)
# This version uses the Intel toolchain
# D. Acreman, October 2019

# Check that we have the required python module
python_version=3.6.3-intel-2017b
have_python_module=`module av Python 2>&1 | grep -c ${python_version}`
if [[ ${have_python_module} -eq 0 ]]; then
    echo "Could not find the module Python/${python_version}"
    echo "Before running this script install python by running: eb Python-${python_version}.eb"
    exit 1
fi

echo "Setting up modules"
module purge
module load Python/${python_version}
module load CMake/3.10.1-GCCcore-6.4.0
module load Bison/3.0.4-GCCcore-6.4.0
module load flex/2.6.4-GCCcore-6.4.0
module list

# Make sure PYTHONPATH is not set otherwise install script will bail out
unset PYTHONPATH

# Tell PETSc to use Intel Math Kernel Library (MKL)
export PETSC_CONFIGURE_OPTIONS="--with-blaslapack-dir=${MKLROOT}/lib/intel64"

# Make mpicc use icc as the back end compiler. PyOP2 uses mpicc with Intel flags.
export I_MPI_CC=icc

echo "Fetching install script"
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install

echo "Installing"
python firedrake-install --disable-ssh --no-package-manager --mpicc mpiicc --mpicxx mpiicpc --mpif90 mpiifort --mpiexec mpirun $@

echo "Done"
