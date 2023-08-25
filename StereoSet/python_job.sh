#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:16gb:1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00

source batch_jobs/_experiment_configuration.sh

echo "Host - $HOSTNAME"
echo "Commit - $(git rev-parse HEAD)"
nvidia-smi

module load python/3.7

virtualenv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies.
cd $HOME/workspace/debias-eval
python -m pip install -e .

# Run code.
python -u "$@" --persistent_dir ${persistent_dir}
