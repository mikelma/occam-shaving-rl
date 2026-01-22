#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:59:00
#SBATCH --output=ppo_mujoco/cpu/baseline_with_layernorm/baseline_with_layernorm_%A_%a.out
#SBATCH --array=0-47

module load python/3.12 mujoco/3.3.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

mkdir $SLURM_TMPDIR/wheels
python -m pip download -d $SLURM_TMPDIR/wheels --no-deps flashbax navix rlax==0.1.6

python -m pip install --no-index --find-links $SLURM_TMPDIR/wheels/ -r requirements/ppo_mujoco/cpu/ppo_mujoco_compute_canada_requirements.txt
python ppo_continuous_action.py --id ${SLURM_ARRAY_TASK_ID} --confs "brax_with_layernorm_baseline_config.bin"
