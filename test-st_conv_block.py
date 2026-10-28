import torch

from layers.st_conv_block import STConvBlock


def count_parameters(model):
    """Count trainable parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("Testing STConvBlock")
    print("=" * 60)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    batch_size = 8
    history = 12
    num_nodes = 113
    in_channels = 11
    out_channels = 64

    # --------------------------------------------------
    # Dummy Input
    # --------------------------------------------------
    x = torch.randn(
        batch_size,
        history,
        num_nodes,
        in_channels,
        requires_grad=True
    )

    # Identity adjacency
    adj = torch.eye(num_nodes)

    # --------------------------------------------------
    # Create Model
    # --------------------------------------------------
    model = STConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        dropout=0.3,
    )

    print(f"Model            : {model.__class__.__name__}")
    print(f"Parameters       : {count_parameters(model):,}")
    print(f"Input Shape      : {tuple(x.shape)}")

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    y = model(x, adj)

    print(f"Output Shape     : {tuple(y.shape)}")

    expected_shape = (
        batch_size,
        history,
        num_nodes,
        out_channels,
    )

    assert y.shape == expected_shape, \
        f"Expected {expected_shape}, got {tuple(y.shape)}"

    print("✓ Forward Pass")

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------
    loss = y.mean()

    loss.backward()

    assert x.grad is not None, \
        "Gradient did not propagate."

    print("✓ Backward Pass")

    # --------------------------------------------------
    # NaN Check
    # --------------------------------------------------
    assert not torch.isnan(y).any(), \
        "Output contains NaN."

    print("✓ NaN Check")

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------
    print()
    print("All STConvBlock tests passed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()