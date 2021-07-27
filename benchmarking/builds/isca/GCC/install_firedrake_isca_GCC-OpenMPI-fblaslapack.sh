#!/bin/bash

# Script for installing Firedrake on Isca (University of Exeter HPC system)
# This version uses the GCC toolchain
# D. Acreman, October 2019

echo "Setting up modules"
module purge
module load Python/3.6.6-foss-2018b
module load CMake/3.13.3-GCCcore-7.3.0
module load Bison/3.0.5-GCCcore-7.3.0
module load flex/2.6.4-GCCcore-7.3.0
module load libxml2/2.9.8-GCCcore-7.3.0
module list

# Make sure PYTHONPATH is not set otherwise install script will bail out
unset PYTHONPATH

# Tell PETSc to download and build BLAS
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"

echo "Fetching install script"
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
echo "Installing"
python3 firedrake-install --disable-ssh --no-package-manager --verbose --mpicc=mpicc --mpicxx=mpicxx --mpif90=mpifort --mpiexec=mpirun $@

echo "Done"
