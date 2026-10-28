import torch

from layers.stsgcl import STSGCL


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing STSGCL")
    print("=" * 60)

    B = 8
    T = 12
    N = 113
    Cin = 64
    Cout = 64

    x = torch.randn(
        B,
        T,
        N,
        Cin,
        requires_grad=True,
    )

    adj = torch.eye(
        3 * N,
    )

    model = STSGCL(
        num_nodes=N,
        input_dim=Cin,
        output_dim=Cout,
        history=T,
        num_gcn_layers=3,
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

    expected = (
        B,
        T - 2,
        N,
        Cout,
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

    print("STSGCL Passed.")

    print("=" * 60)


if __name__ == "__main__":

    main()