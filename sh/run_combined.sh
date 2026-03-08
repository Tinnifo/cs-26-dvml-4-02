!/bin/bash
SBATCH --job-name=gnn-combined
# #SBATCH --partition=prioritized        # <= REMOVE or comment out this line
#SBATCH --account=aau
SBATCH --gres=gpu:1
SBATCH --cpus-per-task=15
SBATCH --mem=24G
SBATCH --time=04:00:00
SBATCH --output=logs/combined_%j.out

cd $HOME/cs-26-dvml-4-02      # adjust if your folder is named differently
source .venv/bin/activate

WANDB_ENTITY=${WANDB_ENTITY:-""}

echo "Running all experiments for all models, budgets and datasets (from run_experiments.py)..."
python3 src/run_experiments.py --use_wandb ${WANDB_ENTITY:+--wandb_entity $WANDB_ENTITY}
