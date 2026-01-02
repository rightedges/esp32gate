import torch
import torchvision.models as models

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Try loading a model and moving to device
try:
    model = models.resnet18(pretrained=False)
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("Test inference shape:", output.shape)
    print("PyTorch is working correctly!")
except Exception as e:
    print("Error during PyTorch test:", e)

