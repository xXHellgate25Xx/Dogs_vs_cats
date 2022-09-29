from torchvision.models import resnet50
import torch
import torch.nn as nn

my_model = resnet50(pretrained = True)

# Modifying Head - classifier

my_model.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
)