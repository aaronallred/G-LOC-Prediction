#!/bin/bash
 
#SBATCH --account=ucb636_asc1
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --partition=amem
#SBATCH --qos=mem
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --job-name=GLOC
#SBATCH --output=cross_validation_rf_implicit.out

module load anaconda
 
conda activate gloc

cd ./scripts/

export PYTHONUNBUFFERED=TRUE
 
python cross_validation_rf_implicit.py
