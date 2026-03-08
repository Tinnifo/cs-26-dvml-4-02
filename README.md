# GNN Reproducibility Project

This project aims to provide a modular and reproducible framework for testing various Graph Neural Networks (GNNs) on Planetoid datasets (Cora, CiteSeer, PubMed).

## Project Structure

- `src/`: Contains model definitions and training logic.
    - `GCN.py`, `GAT.py`, `SAGE.py`, `GIN.py`, `GT.py`: Individual GNN model implementations.
    - `train.py`: Main entry point for training individual models with arguments.
    - `run_experiments.py`: Script to run the full matrix of experiments.
- `eval/`: Contains evaluation scripts and utilities.
    - `evaluation.py`: Metrics and evaluation logic.
    - `Utils.py`: Data utilities like few-shot mask generation.
- `sh/`: Shell scripts to run experiments.
    - `run_gcn.sh`, `run_gat.sh`, etc.: Run individual models.
    - `run_combined.sh`: Runs the full experimental matrix.
- `Dockerfile`: Containerization for reproducibility.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### Local Setup (Virtual Environment)

1. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate the environment:**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Docker Setup

1. **Build the image:**
   ```bash
   docker build -t gnn-experiments .
   ```

2. **Run the full experiment suite:**
   ```bash
   docker run gnn-experiments
   ```

## Running Experiments

### Individual Model Run
To run a specific model with custom parameters:
```bash
python3 src/train.py --model GCN --dataset Cora --budget 20
```

### Full Experimental Matrix
To run all models across all datasets, budgets, and seeds:
```bash
./sh/run_combined.sh
```

## Reproducibility
The framework uses fixed seeds for consistent results. All configurations are centralized in `src/run_experiments.py` and can be customized via arguments in `src/train.py`.
