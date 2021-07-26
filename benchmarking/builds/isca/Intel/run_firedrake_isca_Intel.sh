#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -N Firedrake_Intel
#PBS -l walltime=00:10:00
#PBS -q ptq
# Edit this line to specify a valid project ID
#PBS -A Research_Project-xxxxxx
# Uncomment this line ...
# #PBS -m bae
# ... and put your email here to get notifications
# about jobs beginning, aborting or ending
# #PBS -M my.email@my.uni.ac.uk

# Put the name of your python script here.
# If you want to test the DG advection case the script can be downloaded from
# https://firedrakeproject.org/demos/DG_advection.py
script=DG_advection.py
# Put the location of your firedrake installation here (the directory where install_firedrake_isca_Intel.sh ran)
firedrake_dir=${HOME}/firedrake_builds/Intel

# The following lines should not require modification. The number of nodes and number of
# processors per node are taken from the PBS directive at the top of this script

# Move into the directory the job was submitted from
cd ${PBS_O_WORKDIR}

# Load modules used for Intel builds
module purge
module load Python/3.6.3-intel-2017b

export MYMKLDIR=${MKLROOT}/lib/intel64
export LD_PRELOAD=${MYMKLDIR}/libmkl_def.so:${MYMKLDIR}/libmkl_avx2.so:${MYMKLDIR}/libmkl_core.so:${MYMKLDIR}/libmkl_sequential.so:${MYMKLDIR}/libmkl_intel_lp64.so

# Start firedrake venv
. ${firedrake_dir}/firedrake/bin/activate

# Run the script
echo "Running Firedrake on ${PBS_NP} processors and ${PBS_NUM_NODES} nodes"
mpirun -np ${PBS_NP} python ${script}

echo "All done"
