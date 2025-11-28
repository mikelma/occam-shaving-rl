#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16000M
#SBATCH --time=2:59:00
#SBATCH --output=ppo_atari/ppo_atari_%A_%a.out

module load python/3.10 cuda

virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install  --upgrade pip

python -m pip install -r requirements/updated-atari-cc-reqs.txt
python cleanrl/ppo_atari_envpool_xla_jax_scan.py --env-id Breakout-v5
