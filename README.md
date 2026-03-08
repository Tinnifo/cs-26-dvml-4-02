# GNN Benchmarking on Planetoid Datasets

This project benchmarks various Graph Neural Network (GNN) models on Planetoid datasets (Cora, CiteSeer, PubMed), with integrated **Weights & Biases (W&B)** logging for experiment tracking.

## 1. Prerequisites
Ensure you have the following installed:
- **Python 3.10** or newer.
- **pip** (Python package installer).
- **Docker** (optional, for containerized execution).

## 2. Local Environment Setup

Follow these steps to set up your local development environment from scratch:

### Step 2.1: Create a Virtual Environment
A virtual environment keeps the project's dependencies isolated.
```bash
# Create the environment (using .venv as the name)
python3 -m venv .venv
```

### Step 2.2: Activate the Virtual Environment
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

### Step 2.3: Install Dependencies
Install all required libraries, including PyTorch, PyTorch Geometric, and W&B:
```bash
pip install -r requirements.txt
```

---

## 3. Weights & Biases (W&B) Configuration

Weights & Biases is used for real-time tracking, logging metrics, and visualizing results.

### Step 3.1: Login to W&B
If you don't have an account, sign up at [wandb.ai](https://wandb.ai). Then, log in via the terminal:
```bash
wandb login
```

### Step 3.2: Configure W&B Entity (Optional)
To log to a specific W&B team or username (entity), set the following environment variable:
```bash
export WANDB_ENTITY=your-team-name
```
Alternatively, you can pass this via command-line arguments (see below).

---

## 4. Running Experiments

### Option A: Local Execution
You can run a single model or a full suite of experiments directly from your terminal.

- **Train a Single Model (e.g., GCN on Cora)**:
  ```bash
  python src/train.py --model GCN --dataset Cora --use_wandb
  ```
- **Run the Full Suite (All models, datasets, and budgets)**:
  ```bash
  python src/run_experiments.py --use_wandb
  ```

### Option B: Using Docker
Docker allows you to run the experiments in a consistent, containerized environment.

1. **Build the Docker Image**:
   ```bash
   docker build -t gnn-benchmark .
   ```
2. **Run the Container**:
   To enable W&B logging inside the container, pass your **WANDB_API_KEY** and any other configuration (like `WANDB_ENTITY` or `WANDB_PROJECT`):
   ```bash
   docker run -e WANDB_API_KEY=your_api_key_here \
              -e WANDB_ENTITY=your_entity_name \
              gnn-benchmark
   ```
   *Note: Get your API key from [wandb.ai/settings](https://wandb.ai/settings).*

### Option C: HPC Job Submission (Slurm)
If you are using the **AAU AI-LAB HPC**, use the provided Slurm scripts in the `sh/` directory.

- **Submit a Specific Model Job**:
  ```bash
  sbatch sh/run_gcn.sh
  ```
- **Submit the Entire Experimental Matrix**:
  ```bash
  sbatch sh/run_combined.sh
  ```

---

## 5. Monitoring and Logs

- **W&B Dashboard**: Visit your project on [wandb.ai](https://wandb.ai) to see live training curves (Loss, Accuracy, F1-score) and compare different runs.
- **Local Logs**: If using Slurm, check the `logs/` folder for `.out` files.
- **Interactive Check**: Use `squeue --me` (for Slurm) to check the status of your submitted jobs.

## Project Structure
- `src/`: Core implementation files (`train.py`, `run_experiments.py`, and model architectures).
- `sh/`: Slurm-compatible shell scripts for HPC submission.
- `eval/`: Evaluation metrics and data utility functions.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Configuration for building the container image.
