#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-1:00
#SBATCH --array=1-1
#SBATCH --account=def-mbowling
#SBATCH --gpus-per-node=1

echo "Starting task $SLURM_ARRAY_TASK_ID"
module load python/3.13 gcc opencv mujuco
ls -la

export WANDDB_MODE="offline"

