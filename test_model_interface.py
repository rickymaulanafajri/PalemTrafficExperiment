import torch

from models.dummy_model import DummyModel


def main():

    print("=" * 60)
    print("Testing Model Interface")
    print("=" * 60)

    B = 8
    T = 12
    N = 113
    F = 11

    x = torch.randn(B, T, N, F)

    model = DummyModel(
        history=12,
        horizon=12,
        num_nodes=N,
        in_channels=F,
        hidden_dim=64,
    )

    y = model(x)

    print(f"Model Name   : {model.name}")
    print(f"Parameters   : {model.count_parameters():,}")
    print(f"Input Shape  : {tuple(x.shape)}")
    print(f"Output Shape : {tuple(y.shape)}")

    assert y.shape == (
        B,
        model.horizon,
        N,
    ), "Output shape is incorrect."

    print()
    print("✓ Model interface is correct.")
    print("=" * 60)


if __name__ == "__main__":
    main()