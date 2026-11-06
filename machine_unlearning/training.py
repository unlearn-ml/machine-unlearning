import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils import data

from machine_unlearning import reproducibility
from machine_unlearning import splits

SCHEDULES = ('constant', 'cosine')
Augment = Callable[[torch.Tensor, torch.Generator], torch.Tensor]
Evaluate = Callable[[nn.Module], dict[str, float]]


class PairedBatches:

  def __init__(
      self,
      dataset: data.Dataset,
      split: splits.Split,
      batch_size: int,
      seed: int = 0,
      num_workers: int = 0,
      device: torch.device | str = 'cpu',
  ):
    self.device = torch.device(device)
    self._forget = _shuffled_loader(
        dataset, split.forget, batch_size, seed, 'forget', num_workers, True
    )
    self._retain = _shuffled_loader(
        dataset, split.retain, batch_size, seed, 'retain', num_workers, True
    )
    if min(len(self._forget), len(self._retain)) < 1:
      raise ValueError('The forget and the retain set need at least one batch.')

  def __len__(self) -> int:
    return max(len(self._forget), len(self._retain))

  def __iter__(self) -> Iterator[tuple[tuple[torch.Tensor, ...], ...]]:
    forget_batches = _repeated(self._forget, len(self))
    retain_batches = _repeated(self._retain, len(self))
    for forget_batch, retain_batch in zip(forget_batches, retain_batches):
      yield _to(forget_batch, self.device), _to(retain_batch, self.device)


def train_loader(
    dataset: data.Dataset,
    indices: np.ndarray,
    batch_size: int,
    seed: int = 0,
    num_workers: int = 0,
) -> data.DataLoader:
  return _shuffled_loader(
      dataset, indices, batch_size, seed, 'train', num_workers, False
  )


def unlearn(
    method: Any,
    batches: PairedBatches,
    epochs: int,
    schedule: str = 'constant',
    evaluate: Evaluate | None = None,
) -> list[dict[str, Any]]:
  if epochs < 1:
    raise ValueError(f'epochs must be positive, got {epochs}.')
  if schedule not in SCHEDULES:
    raise ValueError(f'schedule must be one of {SCHEDULES}, got {schedule!r}.')
  model = method.model
  device = next(model.parameters()).device
  if _resolved(batches.device) != _resolved(device):
    raise ValueError(
        f'The batches go to {batches.device} but the model is on {device}.'
    )
  scheduler = None
  if schedule == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        method.optimizer, T_max=len(batches) * epochs
    )
  history = []
  for epoch in range(1, epochs + 1):
    _synchronize(device)
    started = time.perf_counter()
    model.train()
    for forget_batch, retain_batch in batches:
      method.step(forget_batch, retain_batch)
      if scheduler is not None:
        scheduler.step()
    _synchronize(device)
    record = {
        'epoch': epoch,
        'unlearning_seconds': time.perf_counter() - started,
    }
    if evaluate is not None:
      record.update(_evaluation(model, device, evaluate))
    history.append(record)
  model.eval()
  return history


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: data.DataLoader,
    epochs: int,
    milestones: Sequence[int] = (),
    augment: Augment | None = None,
    seed: int = 0,
    evaluate: Evaluate | None = None,
) -> list[dict[str, Any]]:
  if epochs < 1:
    raise ValueError(f'epochs must be positive, got {epochs}.')
  device = next(model.parameters()).device
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(milestones))
  history = []
  for epoch in range(1, epochs + 1):
    model.train()
    generator = reproducibility.generator(seed, 'augment', str(epoch))
    loss_sum = torch.zeros((), device=device)
    samples = 0
    _synchronize(device)
    started = time.perf_counter()
    for inputs, labels in batches:
      if augment is not None:
        inputs = augment(inputs, generator)
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      loss = functional.cross_entropy(model(inputs), labels)
      loss.backward()
      optimizer.step()
      loss_sum += loss.detach() * len(labels)
      samples += len(labels)
    scheduler.step()
    _synchronize(device)
    record = {
        'epoch': epoch,
        'loss': float(loss_sum) / samples,
        'training_seconds': time.perf_counter() - started,
    }
    if evaluate is not None:
      record.update(_evaluation(model, device, evaluate))
    history.append(record)
  model.eval()
  return history


def _shuffled_loader(
    dataset: data.Dataset,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    stream: str,
    num_workers: int,
    drop_last: bool,
) -> data.DataLoader:
  return data.DataLoader(
      data.Subset(dataset, np.asarray(indices).tolist()),
      batch_size=batch_size,
      sampler=reproducibility.SeededShuffle(len(indices), seed, stream),
      drop_last=drop_last,
      num_workers=num_workers,
      generator=reproducibility.generator(seed, stream, 'workers'),
  )


def _repeated(batches: data.DataLoader, count: int) -> Iterator[Any]:
  produced = 0
  while True:
    for batch in batches:
      yield batch
      produced += 1
      if produced == count:
        return


def _evaluation(
    model: nn.Module, device: torch.device, evaluate: Evaluate
) -> dict[str, Any]:
  started = time.perf_counter()
  result = evaluate(model)
  _synchronize(device)
  return {
      'metrics': result,
      'evaluation_seconds': time.perf_counter() - started,
  }


def _resolved(device: torch.device) -> torch.device:
  if device.type == 'cuda' and device.index is None:
    return torch.device('cuda', torch.cuda.current_device())
  return device


def _to(batch: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
  return tuple(item.to(device) for item in batch)


def _synchronize(device: torch.device) -> None:
  if device.type == 'cuda':
    torch.cuda.synchronize(device)
