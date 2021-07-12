#!/bin/bash
#SBATCH --job-name=clr
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=16000

# Load modules
unset LD_LIBRARY_PATH
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.1.243/intel-19.0.3.199

# Activates anaconda environment
source activate pcamenv

# Links the cudnn library from within the conda envs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.conda/envs/pcamenv/lib:$HOME/usr/lib64/

# execution
python3 -u main.py
