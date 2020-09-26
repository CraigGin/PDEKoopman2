#!/bin/bash
#SBATCH -p gpu
#SBATCH -o ./../logs/Sample_Expt.log
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=crgin@ncsu.edu
CUDA_VISIBLE_DEVICES="1" ~/anaconda3/envs/NODE-Operator-TF2/bin/python Sample_Expt.py
