"""
PyTorch Demonstration Script
----------------------------
This script verifies the PyTorch installation in the isolated virtual environment
and demonstrates basic tensor operations, a neural network forward pass, and
the local Transformer Block.
"""

import sys
import torch
import torch.nn as nn


def main():
    print("=" * 60)
    print("           PyTorch Environment Verification & Demo          ")
    print("=" * 60)

    # 1. Python & PyTorch System Info
    print(f"\n[+] Python Version : {sys.version.split()[0]}")
    print(f"[+] Python Path    : {sys.executable}")
    print(f"[+] PyTorch Version: {torch.__version__}")

    # 2. Hardware Acceleration Check
    cuda_available = torch.cuda.is_available()
    print(f"[+] CUDA Available : {cuda_available}")

    if cuda_available:
        device = torch.device("cuda")
        print(f"[+] GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"[+] GPU Device Count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("[i] Device         : CPU")

    # 3. Basic Tensor Operations
    print("\n--- 1. Basic Tensor Operations ---")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    z = torch.matmul(x, y)

    print("Matrix X:")
    print(x)
    print("\nMatrix Y:")
    print(y)
    print("\nMatrix Multiplication (X @ Y):")
    print(z)

    # 4. Neural Network Forward Pass Demo
    print("\n--- 2. Neural Network Forward Pass Demo ---")

    class SimpleClassifier(nn.Module):
        def __init__(self, in_features=4, hidden_features=8, out_features=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features),
                nn.Softmax(dim=-1),
            )

        def forward(self, input_tensor):
            return self.net(input_tensor)

    model = SimpleClassifier().to(device)
    dummy_input = torch.randn(3, 4, device=device)  # Batch of 3 samples, 4 features each
    output = model(dummy_input)

    print("Model Architecture:")
    print(model)
    print(f"\nDummy Input Shape : {dummy_input.shape}")
    print(f"Output Shape      : {output.shape}")
    print("\nOutput Class Probabilities:")
    print(output)

    # 5. Transformer Block Verification (if present)
    try:
        from demo.transformer_block import Block
        print("\n--- 3. Transformer Block Demo ---")
        embedding_dim = 64
        block_size = 16
        n_heads = 4
        transformer = Block(embedding_dim=embedding_dim, block_size=block_size, n_heads=n_heads).to(device)
        sample_tokens = torch.randn(2, block_size, embedding_dim, device=device)
        out_tokens = transformer(sample_tokens)
        print(f"Transformer Input Shape : {sample_tokens.shape}")
        print(f"Transformer Output Shape: {out_tokens.shape}")
    except (ImportError, ModuleNotFoundError):
        pass

    print("\n" + "=" * 60)
    print(" SUCCESS: PyTorch environment is fully ready and working! ")
    print("=" * 60)


if __name__ == "__main__":
    main()
