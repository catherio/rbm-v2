#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=5:00:00
#PBS -l mem=32GB
#PBS -m abe
#PBS -M cao324@nyu.edu
#PBS -N test_rbm_gpu
#PBS -e localhost:$PBS_O_WORKDIR/hpc_output/${PBS_JOBID}_error_${PBS_JOBNAME}
#PBS -o localhost:$PBS_O_WORKDIR/hpc_output/${PBS_JOBID}_output_${PBS_JOBNAME}

module load python/intel/2.7.6
module load cuda/6.5.12

pip install --user Image
pip install --user six
pip install --user timeit

cd /home/cao324/rbm/code

# THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python -B rbm.py
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python -B -c 'from rbm import test_rbm; test_rbm(dataset="vanhateren")'