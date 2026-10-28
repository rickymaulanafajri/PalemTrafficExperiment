import torch

from layers.diffusion_graph_conv import DiffusionGraphConvolution


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing DiffusionGraphConvolution")
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

    # Dummy forward adjacency
    forward_adj = torch.eye(N)

    # Dummy backward adjacency
    backward_adj = torch.eye(N)

    supports = [
        forward_adj,
        backward_adj,
    ]

    model = DiffusionGraphConvolution(
        in_channels=Cin,
        out_channels=Cout,
        diffusion_steps=2,
    )

    print(f"Parameters      : {count_parameters(model):,}")

    y = model(
        x,
        supports,
    )

    print(f"Input Shape     : {tuple(x.shape)}")
    print(f"Output Shape    : {tuple(y.shape)}")

    expected = (
        B,
        T,
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

    print("DiffusionGraphConvolution test passed.")

    print("=" * 60)


if __name__ == "__main__":

    main()