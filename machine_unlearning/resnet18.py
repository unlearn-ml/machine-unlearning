import torch
from torch import nn
from torch.nn import functional


class BasicBlock(nn.Module):

  def __init__(self, in_channels: int, channels: int, stride: int):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels,
        channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(
        channels, channels, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm2d(channels)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels, channels, kernel_size=1, stride=stride, bias=False
          ),
          nn.BatchNorm2d(channels),
      )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = functional.relu(self.bn1(self.conv1(inputs)))
    outputs = self.bn2(self.conv2(outputs))
    return functional.relu(outputs + self.shortcut(inputs))


class ResNet18(nn.Module):

  def __init__(self, num_classes: int):
    super().__init__()
    self.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = _stage(64, 64, stride=1)
    self.layer2 = _stage(64, 128, stride=2)
    self.layer3 = _stage(128, 256, stride=2)
    self.layer4 = _stage(256, 512, stride=2)
    self.linear = nn.Linear(512, num_classes)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = functional.relu(self.bn1(self.conv1(inputs)))
    outputs = self.layer4(self.layer3(self.layer2(self.layer1(outputs))))
    outputs = functional.avg_pool2d(outputs, outputs.shape[-1])
    return self.linear(outputs.flatten(1))


def _stage(in_channels: int, channels: int, stride: int) -> nn.Sequential:
  return nn.Sequential(
      BasicBlock(in_channels, channels, stride),
      BasicBlock(channels, channels, stride=1),
  )
