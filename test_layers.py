import torch

from layers.graph_conv import GraphConvolution
from layers.temporal_conv import TemporalConv
from layers.graph_attention import GraphAttention
from layers.gated_temporal import GatedTemporalConv


def print_result(name, output):
    print(f"{name:<25} {tuple(output.shape)}")


def main():

    print("=" * 60)
    print("Testing Neural Network Layers")
    print("=" * 60)

    # ---------------------------------------------------
    # Dummy Input
    # ---------------------------------------------------
    B = 8
    T = 12
    N = 113
    F = 11

    x = torch.randn(B, T, N, F)

    print(f"Input Shape             {tuple(x.shape)}")
    print()

    # ---------------------------------------------------
    # Identity Adjacency
    # ---------------------------------------------------
    adj = torch.eye(N)

    # ---------------------------------------------------
    # Simple Edge Index
    # Self-loop only
    # ---------------------------------------------------
    edge_index = torch.stack([
        torch.arange(N),
        torch.arange(N)
    ])

    # ===================================================
    # Graph Convolution
    # ===================================================
    graph_conv = GraphConvolution(
        in_channels=11,
        out_channels=32,
    )

    out = graph_conv(x, adj)

    print_result("GraphConvolution", out)

    # ===================================================
    # Temporal Convolution
    # ===================================================
    temporal = TemporalConv(
        in_channels=11,
        out_channels=32,
        kernel_size=3,
    )

    out = temporal(x)

    print_result("TemporalConv", out)

    # ===================================================
    # Gated Temporal
    # ===================================================
    gated = GatedTemporalConv(
        in_channels=11,
        out_channels=32,
    )

    out = gated(x)

    print_result("GatedTemporalConv", out)

    # ===================================================
    # Graph Attention
    # ===================================================
    gat = GraphAttention(
        in_channels=11,
        out_channels=32,
        heads=4,
        concat=False,
    )

    out = gat(x, edge_index)

    print_result("GraphAttention", out)

    print()
    print("=" * 60)
    print("All layer tests completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()