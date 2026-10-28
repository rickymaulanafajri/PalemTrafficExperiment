import torch

from models.stsgcn import STSGCN


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing STSGCN")
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
        requires_grad=True,
    )

    localized_adj = torch.eye(
        3 * N,
    )

    model = STSGCN(
        num_nodes=N,
        input_dim=F,
        history=T,
        horizon=H,
        hidden_dim=64,
        num_layers=2,
        num_gcn_layers=3,
    )

    print(
        f"Model Name : {model.name}"
    )

    print(
        f"Parameters : {count_parameters(model):,}"
    )

    y = model(
        x,
        localized_adj,
    )

    print(
        f"Input Shape  : {tuple(x.shape)}"
    )

    print(
        f"Output Shape : {tuple(y.shape)}"
    )

    expected = (
        B,
        H,
        N,
    )

    assert y.shape == expected

    print("✓ Forward Pass")

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    print("✓ Backward Pass")

    assert not torch.isnan(y).any()

    print("✓ NaN Check")

    print()

    print("STSGCN Passed.")

    print("=" * 60)


if __name__ == "__main__":

    main()