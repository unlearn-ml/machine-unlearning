import pathlib
from collections.abc import Sequence

import torch
import torchvision
from torchvision import transforms

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def load(
    root: str | pathlib.Path, train: bool
) -> torchvision.datasets.CIFAR100:
  return torchvision.datasets.CIFAR100(
      str(root),
      train=train,
      download=True,
      transform=transforms.Compose(
          [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
      ),
  )


def augment(inputs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
  black = transforms.functional.normalize(torch.zeros(3, 1, 1), MEAN, STD)
  return random_crop_and_flip(inputs, 4, black.flatten().tolist(), generator)


def random_crop_and_flip(
    images: torch.Tensor,
    padding: int,
    fill: Sequence[float],
    generator: torch.Generator,
) -> torch.Tensor:
  count, _, height, width = images.shape
  padded = _padded(images, padding, fill)
  rows = torch.randint(2 * padding + 1, (count,), generator=generator)
  columns = torch.randint(2 * padding + 1, (count,), generator=generator)
  flips = torch.rand(count, generator=generator) < 0.5
  crops = []
  for image, row, column, flip in zip(
      padded, rows.tolist(), columns.tolist(), flips.tolist()
  ):
    crop = image[:, row : row + height, column : column + width]
    crops.append(crop.flip(-1) if flip else crop)
  return torch.stack(crops)


def _padded(
    images: torch.Tensor, padding: int, fill: Sequence[float]
) -> torch.Tensor:
  count, channels, height, width = images.shape
  padded = images.new_empty(
      (count, channels, height + 2 * padding, width + 2 * padding)
  )
  padded[:] = images.new_tensor(fill).view(1, channels, 1, 1)
  padded[:, :, padding : padding + height, padding : padding + width] = images
  return padded
