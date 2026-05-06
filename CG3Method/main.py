# -*- coding: utf-8 -*-
import numpy as np
import torch
import time

from funcCNN import (LoadData, preprocess_features, preprocess_adj, processmask,
                     CalCLass01Mat, CalIntraClassMat01, construct_feed_dict_1,
                     arr2sparse)
from CG3Model import GNNModel
from train import HGCN_Model, HGAT_Model
import sys


def tuple_to_torch_sparse(sparse_tuple, device=None):
    """Convert a (coords, values, shape) tuple to a torch sparse_coo_tensor."""
    coords, values, shape = sparse_tuple
    indices = torch.from_numpy(np.asarray(coords).T.astype('int64'))
    values = torch.from_numpy(np.asarray(values, dtype=np.float32))
    t = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
    if device is not None:
        t = t.to(device)
    return t


def main():
    dataset_name = 'cora'
    seed = 123
    hidden_num = 1024
    learning_rate = 0.005
    epochs = 200
    dropout_all = 0.6
    weight_decay = 5e-4

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = LoadData(dataset_name)

    num_classes = np.shape(y_train)[1]
    num_inst = np.shape(y_train)[0]
    features = preprocess_features(features)
    feature_sp = tuple_to_torch_sparse(features, device=device)

    input_dim = features[2][1]
    support = preprocess_adj(adj)
    support_sp = tuple_to_torch_sparse(support, device=device)
    num_inst = features[2][0]

    trtemask = processmask(train_mask)

    # ---- placeholders dict (mutable; updated each step) ----
    placeholders = {
        'support': support_sp,
        'features': feature_sp,
        'labels': torch.zeros((num_inst, y_train.shape[1]), dtype=torch.float32, device=device),
        'labels_mask': torch.zeros(num_inst, dtype=torch.int32, device=device),
        'dropout': 0.0,
        'num_features_nonzero': features[1].shape[0],
    }

    paras = dict()
    paras['hidden_num'] = hidden_num
    paras['weight_decay'] = weight_decay
    paras['dataset'] = dataset_name
    paras['seed1'] = seed
    paras['seed2'] = seed
    

    y_dim1 = np.argmax(y_train, axis=1)
    y_dim = np.ones([num_inst]) * -1
    tr_idx = np.argwhere(np.sum(y_train, axis=1) > 0)[:, 0]
    y_dim[tr_idx] = y_dim1[tr_idx]

    intra_class_idx = []
    for i in range(num_classes):
        intra_class_idx.append(np.argwhere(y_dim == i)[:, 0])

    train_mat01 = CalCLass01Mat(y_train, train_mask)
    mats_intra_inter = CalIntraClassMat01(y_dim1[tr_idx])
    num_labeled = int(np.sum(y_train))
    mats_intra_inter[0] += np.eye(num_labeled)
    

    # Reset seeds before model construction (matches the TF code's ordering).
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # The dropout getter / num_features_nonzero getter are kept in dp_fea0 so
    # CG3Layer.GraphConvolution can read the current values at forward time.
    dp_fea0 = [
        lambda: placeholders['dropout'],
        placeholders['num_features_nonzero'],
    ]
    
    experiments = [
        {"name": "gcn_hgcn", "local_model": "gcn", "global": "hgcn"},
        {"name": "gat_hgcn", "local_model": "gat", "global": "hgcn"},
        {"name": "gcn_hgat", "local_model": "gcn", "global": "hgat"},
        {"name": "gat_hgat", "local_model": "gat", "global": "hgat"},
    ]
    
    results = {}
    
    y_train_t = torch.from_numpy(y_train.astype('float32')).to(device)
    y_val_t = torch.from_numpy(y_val.astype('float32')).to(device)
    y_test_t = torch.from_numpy(y_test.astype('float32')).to(device)
    train_mask_t = torch.from_numpy(train_mask.astype('int32')).to(device)
    val_mask_t = torch.from_numpy(val_mask.astype('int32')).to(device)
    test_mask_t = torch.from_numpy(test_mask.astype('int32')).to(device)
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")

        # reset seeds for fair comparison
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # fresh HGCN for each experiment
        if exp["name"].endswith("hgat"):
            global_model = HGAT_Model(placeholders, paras, adj=adj)
        else:
            global_model = HGCN_Model(placeholders, paras, adj=adj)

        global_model = global_model.to(device)


        # fresh local model for each experiment
        model = GNNModel(
            learning_rate=learning_rate,
            num_classes=num_classes,
            h=hidden_num,
            input_dim=input_dim,
            global_model = global_model,
            train_idx=tr_idx,
            trtemask=trtemask,
            dp_fea0=dp_fea0,
            edge_pos=support[0],
            train_mat01=train_mat01,
            mat01_tr_te=mats_intra_inter,
            weight_decay=weight_decay,
            local_model=exp["local_model"]
        ).to(device)

        test_accs = []
        val_accs = []
        train_accs = []

        for epoch in range(epochs):
            # ---------------- train ----------------
            model.train()
            global_model.train()
            placeholders["dropout"] = dropout_all

            model.optimizer.zero_grad()

            outputs, loss, accuracy = model(
                feature_sp,
                support_sp,
                y_train_t,
                train_mask_t
            )

            loss.backward()
            model.optimizer.step()

            train_acc_val = float(accuracy.detach().cpu())

            # ---------------- eval ----------------
            model.eval()
            global_model.eval()
            placeholders["dropout"] = 0.0

            with torch.no_grad():
                _, test_loss, test_acc = model(
                    feature_sp,
                    support_sp,
                    y_test_t,
                    test_mask_t
                )

                _, val_loss, val_acc = model(
                    feature_sp,
                    support_sp,
                    y_val_t,
                    val_mask_t
                )

            test_acc_val = float(test_acc.detach().cpu())
            val_acc_val = float(val_acc.detach().cpu())

            train_accs.append(train_acc_val)
            test_accs.append(test_acc_val)
            val_accs.append(val_acc_val)

            print(
                f"{exp['name']} | "
                f"Epoch {epoch+1:04d} | "
                f"train={train_acc_val:.5f} "
                f"test={test_acc_val:.5f} "
                f"val={val_acc_val:.5f}"
            )

        # choose test accuracy at best validation epoch
        best_epoch = np.argmax(val_accs)

        results[exp["name"]] = {
            "best_epoch": int(best_epoch + 1),
            "best_val": float(val_accs[best_epoch]),
            "best_test": float(test_accs[best_epoch]),
            "max_test": float(np.max(test_accs)),
        }

    print("\n========== FINAL RESULTS ==========")
    for name, result in results.items():
        print(name)
        print(result)
        print()
    
if __name__ == "__main__":
    main()