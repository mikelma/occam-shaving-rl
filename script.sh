#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-1:00
#SBATCH --array=1-1
#SBATCH --account=def-mbowling
#SBATCH --gpus-per-node=1

echo "Starting task $SLURM_ARRAY_TASK_ID"
module load python/3.13 gcc opencv mujuco

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip install --no-index -r compute-canada-requirements.txt

export WANDDB_MODE="offline"

python ppo_continuous_action.py
