import math
from typing import NamedTuple

import numpy as np
import torch
from sklearn import model_selection
from sklearn import svm
from torch import nn
from torch.utils import data

from machine_unlearning import splits

DEFAULT_SUBSAMPLE_SEED = 42


class Outputs(NamedTuple):
  logits: torch.Tensor
  labels: torch.Tensor


def evaluate(
    model: nn.Module,
    train_set: data.Dataset,
    test_set: data.Dataset,
    split: splits.Split,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, float]:
  forget = outputs(model, train_set, split.forget, batch_size, num_workers)
  retain = outputs(model, train_set, split.retain, batch_size, num_workers)
  test = outputs(model, test_set, split.test, batch_size, num_workers)
  return {
      'retain_accuracy': accuracy(retain),
      'forget_accuracy': accuracy(forget),
      'test_accuracy': accuracy(test),
      'svc_membership_efficacy': svc_membership_efficacy(retain, test, forget),
  }


@torch.no_grad()
def outputs(
    model: nn.Module,
    dataset: data.Dataset,
    indices: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Outputs:
  loader = data.DataLoader(
      data.Subset(dataset, np.asarray(indices).tolist()),
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      generator=torch.Generator(),
  )
  device = next(model.parameters()).device
  was_training = model.training
  model.eval()
  logits = []
  labels = []
  try:
    for inputs, targets in loader:
      logits.append(model(inputs.to(device)).cpu())
      labels.append(targets)
  finally:
    model.train(was_training)
  return Outputs(torch.cat(logits), torch.cat(labels))


def accuracy(result: Outputs) -> float:
  correct = int((result.logits.argmax(dim=1) == result.labels).sum())
  return 100 * float(correct) / len(result.labels)


def svc_membership_efficacy(
    retain: Outputs,
    test: Outputs,
    forget: Outputs,
    seed: int = DEFAULT_SUBSAMPLE_SEED,
) -> float:
  if not all(
      bool(torch.isfinite(result.logits).all())
      for result in (retain, test, forget)
  ):
    return math.nan
  size = min(len(retain.labels), len(test.labels))
  members = _stratified_subset(retain, size, seed)
  nonmembers = _stratified_subset(test, size, seed)
  features = _true_class_probability(
      torch.cat([members.logits, nonmembers.logits]),
      torch.cat([members.labels, nonmembers.labels]),
  )
  membership = np.concatenate(
      [
          np.ones(len(members.labels), dtype=np.int64),
          np.zeros(len(nonmembers.labels), dtype=np.int64),
      ]
  )
  classifier = svm.SVC(C=3, gamma='auto', kernel='rbf')
  classifier.fit(features, membership)
  predictions = classifier.predict(
      _true_class_probability(forget.logits, forget.labels)
  )
  nonmembers_predicted = np.count_nonzero(predictions == 0)
  return 100 * float(nonmembers_predicted) / len(predictions)


def _stratified_subset(result: Outputs, size: int, seed: int) -> Outputs:
  if size == len(result.labels):
    return result
  splitter = model_selection.StratifiedShuffleSplit(
      n_splits=1, test_size=size, random_state=seed
  )
  _, chosen = next(
      splitter.split(np.zeros(len(result.labels)), result.labels.numpy())
  )
  return Outputs(result.logits[chosen], result.labels[chosen])


def _true_class_probability(
    logits: torch.Tensor, labels: torch.Tensor
) -> np.ndarray:
  probabilities = torch.softmax(logits, dim=1)
  return torch.gather(probabilities, 1, labels[:, None]).numpy()
