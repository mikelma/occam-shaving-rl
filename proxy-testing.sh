#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-2:59
#SBATCH --array=1-2
#SBATCH --account=def-mbowling
#SBATCH --gpus-per-node=1

echo "Starting task $SLURM_ARRAY_TASK_ID"
module load python/3.12 gcc opencv

echo "Loading Keys"
source keys.env

echo "Starting Proxy"

pip install --no-index requests[socks]
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi

# wandb setup

# wandb offline

echo "Logging into Wandb..."
wandb login $WANDB_API_KEY

# Create venv in $SLURM_TMPDIR
VENV_DIR=$SLURM_TMPDIR/venv
uv venv "$VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"

# Install Requirements
uv sync --active 
uv run --active python ppo_continuous_action.py --shared_network --seed=$SLURM_ARRAY_TASK_ID --wandb_project_name=testing
