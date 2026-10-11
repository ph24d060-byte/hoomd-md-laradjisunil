#!/bin/bash

#SBATCH --job-name=ls_gpu_test
#SBATCH --partition=sgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=ls_gpu_test_%j.out

set -euo pipefail

module purge
module load nvhpc-24.5/nvhpc-hpcx-cuda12/24.5

# Activate the same environment used by your membrane simulations.
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PATH="$HOME/micromamba/bin:$PATH"

eval "$(micromamba shell hook -s bash)"
micromamba activate hoomd_env

# Match your existing HOOMD/CUDA runtime environment.
export NVHPC_ROOT=/home/apps/nvhpc/Linux_x86_64/24.5

export PYTHONPATH="/scratch/ph24d060/hoomd-install${PYTHONPATH:+:$PYTHONPATH}"

export LD_LIBRARY_PATH="$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/compilers/lib64:$NVHPC_ROOT/cuda/lib64:$NVHPC_ROOT/math_libs/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "===== GPU information ====="
nvidia-smi

echo "===== Python and HOOMD ====="
which python

python -c "import hoomd; print(hoomd.__file__); print('GPU:', hoomd.version.gpu_enabled)"

echo "===== Laradji-Sunil GPU test ====="
python gpu_smoke_test.py
