#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-1:00
#SBATCH --array=1-1
#SBATCH --account=def-mbowling
#SBATCH --gpus-per-node=1

echo "Starting task $SLURM_ARRAY_TASK_ID"
module load python/3.13 gcc opencv mujuco

# Set up variables
VENV_SOURCE=".venv"
VENV_TARGET="$SLURM_TMPDIR/.venv"

echo "Copying virtual environment to local scratch..."
cp -r "$VENV_SOURCE" "$VENV_TARGET"

# Point Python to use the copied virtualenv
source "$VENV_TARGET/bin/activate"

wandb offline

python ppo_continuous_action.py
