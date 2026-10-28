import torch

from layers.adaptive_graph_conv import AdaptiveGraphConvolution


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing AdaptiveGraphConvolution")
    print("=" * 60)

    B = 8
    T = 12
    N = 113
    Cin = 11
    Cout = 64

    x = torch.randn(
        B,
        T,
        N,
        Cin,
        requires_grad=True,
    )

    model = AdaptiveGraphConvolution(
        num_nodes=N,
        in_channels=Cin,
        out_channels=Cout,
        embedding_dim=10,
    )

    print(f"Parameters      : {count_parameters(model):,}")

    y = model(x)

    print(f"Input Shape     : {tuple(x.shape)}")
    print(f"Output Shape    : {tuple(y.shape)}")

    assert y.shape == (
        B,
        T,
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

    print("AdaptiveGraphConvolution test passed.")

    print("=" * 60)


if __name__ == "__main__":

    main()