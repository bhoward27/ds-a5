#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=slow
#SBATCH --ntasks=4
#SBATCH --mem=10G

python /scratch/assignment5/test_scripts/triangle_counting_tester.pyc --execPath=$HOME/cmpt-431/assignments/5/ds-a5/triangle_counting_parallel