
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataloader
print("testing")

# define model

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x



# tranining loop


for dataset_name in DATASETS:
    print(f"\n========== DATASET: {dataset_name} ==========")

    dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    base_data = dataset[0].to(device)

    for model_name in MODELS:
        print(f"\n---- Model: {model_name} ----")

        for budget in BUDGETS:
            all_metrics = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                data = base_data.clone()
                data = set_few_label_mask(data, budget, seed)

                model = get_model(model_name, dataset).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                # Training
                for epoch in range(EPOCHS):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = F.cross_entropy(
                        out[data.train_mask],
                        data.y[data.train_mask]
                    )
                    loss.backward()
                    optimizer.step()

                metrics = evaluate(model, data)
                all_metrics.append(metrics)

            all_metrics = np.array(all_metrics)
            mean_metrics = np.mean(all_metrics, axis=0)
            std_metrics = np.std(all_metrics, axis=0)

            print(
                f"Budget {budget} | "
                f"Acc={mean_metrics[0]:.4f}±{std_metrics[0]:.4f} | "
                f"MacroF1={mean_metrics[3]:.4f}±{std_metrics[3]:.4f}"
            )



# Default value
