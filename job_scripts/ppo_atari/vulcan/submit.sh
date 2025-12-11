#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16000M
#SBATCH --time=8:59:00
#SBATCH --output=ppo_atari/ppo_atari_%A_%a.out

module load python/3.11 cuda opencv/4.10 swig/4.1

virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install  --upgrade pip

python -m pip install -r requirements/updated-atari-cc-reqs.txt
python cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id Breakout-v5
