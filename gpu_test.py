import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Create tensors on GPU
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# Matrix multiplication
z = torch.matmul(x, y)

print("Matrix multiplication successful!")
print("Result shape:", z.shape)

if device.type == "cuda":
    print(
        "GPU Memory Allocated:",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
        "MB"
    )

print("GPU Test Completed!")