#!/bin/bash
echo "Running all GNN experiments..."

for model in GCN GAT SAGE GIN GT
do
    echo "------------------------------------------------"
    echo "Running $model..."
    python3 src/train.py --model $model --dataset Cora --budget 20
done
