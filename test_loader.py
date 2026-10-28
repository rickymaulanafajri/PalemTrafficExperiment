from datasets.dataloader import load_dataset

train_loader, val_loader, test_loader, scaler = load_dataset(
    "data/transgraphai_data_30days_multivariate.npz",
    history=12,
    horizon=12,
)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

x, y = next(iter(train_loader))

print(x.shape)
print(y.shape)