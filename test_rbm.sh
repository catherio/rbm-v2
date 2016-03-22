#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -m abe
#PBS -M cao324@nyu.edu
#PBS -N test_rbm

module load python/intel/2.7.6
pip install --user Image
pip install --user six
pip install --user timeit

cd /home/cao324/rbm/

python rbm.py
