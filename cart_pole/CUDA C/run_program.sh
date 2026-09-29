#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=run_my_program
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=students
#SBATCH --output=run_my_program.out
#SBATCH --mail-user=aarya03@student.ubc.ca
#SBATCH --mail-type=ALL

# Load necessary modules if required
# module load cuda/11.2

# Update PATH and LD_LIBRARY_PATH if necessary
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Run your executable
./run -log 0
# Optionally, include any other commands you need
