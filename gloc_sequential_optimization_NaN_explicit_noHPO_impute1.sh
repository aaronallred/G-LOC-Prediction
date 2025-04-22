#!/bin/bash
 
#SBATCH --account=ucb636_asc1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=amem
#SBATCH --qos=mem
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --job-name=GLOC
#SBATCH --output=GLOC_sequential_optimization_NaN_explicit_noHPO_impute1.out

module load anaconda
 
conda activate gloc

cd ./scripts/

export PYTHONUNBUFFERED=TRUE
 
python sequential_optimization_NaN_explicit_noHPO_impute1.py
