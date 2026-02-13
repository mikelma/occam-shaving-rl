#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:59:00
#SBATCH --output=ppo_brax/cpu/brax_small_sweep/job_%A_%a.out

# Calculate the actual ID for your python script
# SLURM_ARRAY_TASK_ID (0-999) + OFFSET (0, 1000, 2000...)
REAL_ID=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

echo "Rank ID: $SLURM_ARRAY_TASK_ID"
echo "Offset: $OFFSET"
echo "Running Application ID: $REAL_ID"

module load python/3.12 mujoco/3.3.0

# Setup environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

# Install dependencies
mkdir -p $SLURM_TMPDIR/wheels
python -m pip download -d $SLURM_TMPDIR/wheels --no-deps flashbax navix rlax==0.1.6
python -m pip install --no-index --find-links $SLURM_TMPDIR/wheels/ -r requirements/ppo_brax/cpu/ppo_brax_compute_canada_requirements.txt

# Run the task with the calculated REAL_ID
python ppo_continuous_action.py --id ${REAL_ID} --confs "brax_small_config.bin"
