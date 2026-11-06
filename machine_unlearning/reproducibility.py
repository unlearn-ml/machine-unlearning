import hashlib
import importlib.metadata
import pathlib
import platform
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils import data

_SEED_MASK = 2**63 - 1
_PACKAGES = (
    'machine-unlearning',
    'numpy',
    'scikit-learn',
    'torch',
    'torchvision',
)


def make_deterministic() -> None:
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cuda.matmul.allow_tf32 = False


def generator(seed: int, *names: str) -> torch.Generator:
  return torch.Generator().manual_seed(_derived_seed(seed, *names))


def manual_seed(seed: int, *names: str) -> None:
  torch.manual_seed(_derived_seed(seed, *names))


def _derived_seed(seed: int, *names: str) -> int:
  text = '/'.join([str(seed), *names]).encode('utf-8')
  digest = hashlib.sha256(text).digest()
  return int.from_bytes(digest[:8], 'little') & _SEED_MASK


class SeededShuffle(data.Sampler[int]):

  def __init__(self, size: int, seed: int, stream: str):
    super().__init__()
    self._size = size
    self._seed = seed
    self._stream = stream
    self._passes = 0

  def __len__(self) -> int:
    return self._size

  def __iter__(self) -> Iterator[int]:
    order = generator(self._seed, self._stream, str(self._passes))
    self._passes += 1
    return iter(torch.randperm(self._size, generator=order).tolist())


def fingerprint(value: pathlib.Path | np.ndarray) -> str:
  if isinstance(value, pathlib.Path):
    with value.open('rb') as f:
      return hashlib.file_digest(f, 'sha256').hexdigest()
  indices = np.ascontiguousarray(value, dtype=np.int64)
  return hashlib.sha256(indices.tobytes()).hexdigest()


def environment(device: torch.device | str) -> dict[str, Any]:
  device = torch.device(device)
  record = {name: importlib.metadata.version(name) for name in _PACKAGES}
  record['python'] = platform.python_version()
  record['cuda'] = torch.version.cuda
  record['cudnn'] = torch.backends.cudnn.version()
  if device.type == 'cuda':
    record['device'] = torch.cuda.get_device_name(device)
  else:
    record['device'] = platform.processor() or platform.machine()
  return record
