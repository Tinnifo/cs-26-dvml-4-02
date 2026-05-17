#!/bin/bash
#SBATCH --job-name=cg3perclass
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 1. Move to the directory where you submitted the job
set -euo pipefail

cd /ceph/home/student.aau.dk/ab10ix/cs-26-dvml-4-02 || exit 1
mkdir -p logs

# 2. Load necessary cluster modules (Ask your admin for the exact names)
# module load cuda/12.1

# 3. Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found in $(pwd)"
    exit 1
fi


export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_LAUNCH_BLOCKING=1

# 4. Debug info - This will show up in your .out log
echo "Working directory: $(pwd)"
echo "Using python: $(which python)"
nvidia-smi


# 5. Run the job
# Note: Added quotes around the budget list for Hydra safety

python src/train.py --multirun +experiment=cg3_combinations device=cuda