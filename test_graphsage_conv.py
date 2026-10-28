import torch

from layers.graphsage_conv import GraphSAGEConv


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing GraphSAGEConv")
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

    # Simple self-loop graph
    edge_index = torch.stack(
        [
            torch.arange(N),
            torch.arange(N),
        ]
    )

    model = GraphSAGEConv(
        in_channels=Cin,
        out_channels=Cout,
        aggr="mean",
    )

    print(f"Parameters      : {count_parameters(model):,}")

    y = model(
        x,
        edge_index,
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
    print("GraphSAGEConv Passed.")
    print("=" * 60)


if __name__ == "__main__":

    main()