import torch

from layers.stsgcm import STSGCM


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing STSGCM")
    print("=" * 60)

    N = 113

    Cin = 64

    Cout = 64

    x = torch.randn(
        3 * N,
        Cin,
        requires_grad=True,
    )

    adj = torch.eye(
        3 * N,
    )

    model = STSGCM(
        num_nodes=N,
        in_channels=Cin,
        out_channels=Cout,
        num_layers=3,
        activation="glu",
    )

    print(
        f"Parameters : {count_parameters(model):,}"
    )

    y = model(
        x,
        adj,
    )

    print(
        f"Input Shape  : {tuple(x.shape)}"
    )

    print(
        f"Output Shape : {tuple(y.shape)}"
    )

    assert y.shape == (
        N,
        Cout,
    )

    print("✓ Forward Pass")

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    print("✓ Backward Pass")

    assert not torch.isnan(y).any()

    print("✓ NaN Check")

    print()

    print("STSGCM Passed.")

    print("=" * 60)


if __name__ == "__main__":

    main()