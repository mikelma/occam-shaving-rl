#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16000M
#SBATCH --time=02:59:00
#SBATCH --output=job_testing/job_testing_%A_%a.out

module load python/3.12 mujoco/3.3.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip download -d $SLURM_TMPDIR/wheels --no-deps flashbax navix rlax==0.1.6

python -m pip install -U --no-index --find-links $SLURM_TMPDIR/wheels/ -r updated-reqs.txt
python ppo_continuous_action.py
