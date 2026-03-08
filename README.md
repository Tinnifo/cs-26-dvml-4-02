# GNN Benchmarking on Planetoid Datasets (AAU AI-LAB HPC)

This project benchmarks various Graph Neural Network (GNN) models on Planetoid datasets (Cora, CiteSeer), optimized for the **AAU AI-LAB HPC** environment with integrated **Weights & Biases (W&B)** logging.

## Project Structure

- **`src/`**: Core implementation files.
  - `train.py`: Primary script for training a single model with configurable parameters and W&B logging.
  - `run_experiments.py`: Script to run a comprehensive suite of experiments across multiple models, datasets, and label budgets.
  - `GCN.py`, `GAT.py`, `SAGE.py`, `GIN.py`, `GT.py`: Individual model architectures.
- **`sh/`**: Slurm-compatible shell scripts for HPC job submission.
  - `run_gcn.sh`, `run_gat.sh`, etc.: Submit single-model training jobs.
  - `run_all.sh`: Sequentially trains all models for a specific dataset/budget in one job.
  - `run_combined.sh`: Runs the full experimental matrix (all models, budgets, and datasets).
- **`eval/`**: Evaluation metrics and data utility functions.
- **`logs/`**: Directory for Slurm output and error files.
- **`requirements.txt`**: Python dependencies, including `wandb`.

## Setup & Weights & Biases Logging

The experiments are logged to Weights & Biases for real-time tracking and comparison.

1. **Login to W&B**: Before running jobs on the HPC, ensure you are logged in:
   ```bash
   wandb login
   ```
2. **Specify Team (Entity)**: To log to a specific W&B team/entity, set the `WANDB_ENTITY` environment variable:
   ```bash
   export WANDB_ENTITY=your-team-name
   ```
   If not set, runs will log to your personal account.

## Running Experiments on HPC (AI-LAB)

The shell scripts are configured for the `gpu` partition with appropriate resource requests.

### 1. Individual Model Runs
To train a specific model (e.g., GCN) on the default dataset (Cora):
```bash
sbatch sh/run_gcn.sh
```

### 2. Sequential Runs (Multiple Models)
To train all models (GCN, GAT, SAGE, GIN, GT) sequentially in a single GPU allocation:
```bash
sbatch sh/run_all.sh
```

### 3. Full Combined Matrix
To run the complete benchmark (multiple datasets and label budgets) as defined in `run_experiments.py`:
```bash
sbatch sh/run_combined.sh
```

## Monitoring
- **Slurm Status**: Use `squeue --me` to check your job status.
- **Logs**: Check `logs/` for standard output (e.g., `cat logs/gcn_JOBID.out`).
- **W&B Dashboard**: Visit [wandb.ai](https://wandb.ai) to view live training curves, metrics (Accuracy, F1-score), and hardware utilization.
