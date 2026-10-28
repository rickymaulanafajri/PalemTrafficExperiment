import torch

from models.stgcn import STGCN


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing STGCN")
    print("=" * 60)

    B = 8
    T = 12
    N = 113
    F = 11

    H = 12

    x = torch.randn(
        B,
        T,
        N,
        F,
        requires_grad=True
    )

    adj = torch.eye(N)

    model = STGCN(
        num_nodes=N,
        input_dim=F,
        history=T,
        horizon=H,
        hidden_dim=64,
        num_blocks=2,
    )

    print(f"Model Name      : {model.name}")
    print(f"Parameters      : {count_parameters(model):,}")

    y = model(x, adj)

    print(f"Input Shape     : {tuple(x.shape)}")
    print(f"Output Shape    : {tuple(y.shape)}")

    expected = (B, H, N)

    assert y.shape == expected

    print("✓ Forward Pass")

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    print("✓ Backward Pass")

    assert not torch.isnan(y).any()

    print("✓ NaN Check")

    print()
    print("STGCN test passed.")
    print("=" * 60)


if __name__ == "__main__":

    main()