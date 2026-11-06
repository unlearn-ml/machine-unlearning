import operator
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class Split(NamedTuple):
  forget: np.ndarray
  retain: np.ndarray
  test: np.ndarray


def random_forgetting(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    ratio: float,
    seed: int,
) -> Split:
  train_labels = np.asarray(train_labels)
  seed = operator.index(seed)
  if not 0.0 < ratio < 1.0:
    raise ValueError(f'ratio must be in (0, 1), got {ratio}.')
  count = int(len(train_labels) * ratio)
  forget = _same_share_of_every_class(train_labels, count, seed)
  if forget.size == 0:
    raise ValueError(f'The ratio {ratio} selects no sample.')
  return Split(
      forget=forget,
      retain=np.setdiff1d(np.arange(len(train_labels)), forget),
      test=np.arange(len(test_labels)),
  )


def _same_share_of_every_class(
    labels: np.ndarray, count: int, seed: int
) -> np.ndarray:
  classes_in_order = list(dict.fromkeys(labels.tolist()))
  per_class = count // len(classes_in_order)
  chosen = []
  for label in classes_in_order:
    candidates = np.flatnonzero(labels == label)
    if per_class > len(candidates):
      raise ValueError(
          f'Class {label} has {len(candidates)} samples, fewer than the'
          f' {per_class} to forget from every class.'
      )
    random_state = np.random.RandomState(seed)
    chosen.append(random_state.choice(candidates, per_class, replace=False))
  return np.concatenate(chosen)
