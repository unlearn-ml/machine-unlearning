import argparse
import itertools
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils import data

from machine_unlearning import cifar100
from machine_unlearning import metrics
from machine_unlearning import reproducibility
from machine_unlearning import resnet18
from machine_unlearning import rosu
from machine_unlearning import splits
from machine_unlearning import training

_HUGGING_FACE = 'https://huggingface.co/ResearchBrother/machine-unlearning'
_SHARED_MODELS = {
    'original.pt': (
        'f4b2c44061599da3993f4180504bd81ffae7c9d51d614c526701cf161b6054a8'
    ),
    'retrain_ratio_0.1_seed_42.pt': (
        '49c3b7e7cffc62018f7dbc3b10594a4ffbc08f7b59e93c9ffbee4c5396f33ccb'
    ),
    'retrain_ratio_0.1_seed_128.pt': (
        '0eb70ba825d67ec520d21ef7273acd3b490332b86c5ee134ac06a63f506f002e'
    ),
    'retrain_ratio_0.1_seed_199.pt': (
        '5dca5c402688ba63683808fe58597b96c8e759084b4ddde1a6810a783cee6715'
    ),
}


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(prog='mu')
  commands = parser.add_subparsers(dest='command', required=True)
  _add_train_command(commands)
  _add_unlearn_command(commands)
  _add_evaluate_command(commands)
  args = parser.parse_args(argv)
  args.run(args)


def _add_train_command(commands: Any) -> None:
  parser = commands.add_parser('train')
  _add_data_flags(parser)
  parser.add_argument('--ratio', type=float)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--learning-rate', type=float, default=0.1)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight-decay', type=float, default=5e-4)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--milestones', type=int, nargs='*', default=[100, 150])
  parser.add_argument('--run-seed', type=int, default=0)
  parser.add_argument('--save-model', type=pathlib.Path, required=True)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.set_defaults(run=_train, parser=parser)


def _add_unlearn_command(commands: Any) -> None:
  parser = commands.add_parser('unlearn')
  _add_data_flags(parser)
  _add_forget_flags(parser)
  parser.add_argument('--checkpoint', type=pathlib.Path)
  parser.add_argument('--learning-rate', type=float, default=0.01566)
  parser.add_argument('--amplification-scale', type=float, default=1.676)
  parser.add_argument('--perturbation-radius', type=float, default=1e-3)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--weight-decay', type=float, default=5e-4)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument(
      '--schedule', choices=training.SCHEDULES, default='cosine'
  )
  parser.add_argument('--run-seed', type=int, default=0)
  parser.add_argument('--save-model', type=pathlib.Path)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.set_defaults(run=_unlearn, parser=parser)


def _add_evaluate_command(commands: Any) -> None:
  parser = commands.add_parser('evaluate')
  _add_data_flags(parser)
  _add_forget_flags(parser)
  parser.add_argument('--checkpoint', type=pathlib.Path)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.set_defaults(run=_evaluate, parser=parser)


def _add_data_flags(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--data-root', type=pathlib.Path, default='data')
  parser.add_argument('--num-workers', type=int, default=4)
  parser.add_argument(
      '--device', default='cuda' if torch.cuda.is_available() else 'cpu'
  )


def _add_forget_flags(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--ratio', type=float, required=True)
  parser.add_argument('--seed', type=int, required=True)


def _train(args: argparse.Namespace) -> None:
  if (args.ratio is None) != (args.seed is None):
    args.parser.error('Give --ratio and --seed together.')
  _prepare_outputs(args.parser, args.save_model, args.output)
  environment = reproducibility.environment(args.device)
  reproducibility.make_deterministic()
  train_set = cifar100.load(args.data_root, train=True)
  test_set = cifar100.load(args.data_root, train=False)
  train_indices = np.arange(len(train_set))
  if args.ratio is not None:
    train_indices = _split(args, train_set, test_set).retain
  test_indices = np.arange(len(test_set))
  batches = training.train_loader(
      train_set, train_indices, args.batch_size, args.run_seed, args.num_workers
  )
  reproducibility.manual_seed(args.run_seed, 'model')
  model = resnet18.ResNet18(num_classes=len(train_set.classes))
  model.to(args.device)
  optimizer = torch.optim.SGD(
      model.parameters(),
      lr=args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay,
  )
  history = training.train(
      model,
      optimizer,
      batches,
      args.epochs,
      args.milestones,
      augment=cifar100.augment,
      seed=args.run_seed,
      evaluate=_with_progress(
          _test_accuracy(test_set, test_indices, args.num_workers)
      ),
  )
  _save_model(args.save_model, model)
  _write(
      args.output,
      {
          'command': 'train',
          'settings': {
              'learning_rate': args.learning_rate,
              'momentum': args.momentum,
              'weight_decay': args.weight_decay,
              'batch_size': args.batch_size,
              'epochs': args.epochs,
              'milestones': args.milestones,
          },
          'forget': _forget(args),
          'sizes': {'train': len(train_indices), 'test': len(test_indices)},
          'run_seed': args.run_seed,
          'epochs': history,
          'fingerprints': {
              'model': reproducibility.fingerprint(args.save_model),
              'train': reproducibility.fingerprint(train_indices),
              'test': reproducibility.fingerprint(test_indices),
          },
          'environment': environment,
      },
  )
  print(json.dumps(history[-1], indent=2, sort_keys=True))


def _unlearn(args: argparse.Namespace) -> None:
  checkpoint = args.checkpoint or _hugging_face_model(
      args.parser, 'original.pt'
  )
  _check_inputs(args.parser, checkpoint)
  _prepare_outputs(args.parser, args.output, args.save_model)
  environment = reproducibility.environment(args.device)
  reproducibility.make_deterministic()
  train_set = cifar100.load(args.data_root, train=True)
  test_set = cifar100.load(args.data_root, train=False)
  split = _split(args, train_set, test_set)
  model = _load_model(checkpoint, len(train_set.classes), args.device)
  reproducibility.manual_seed(args.run_seed, 'method')
  optimizer = torch.optim.SGD(
      model.parameters(),
      lr=args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay,
  )
  method = rosu.ROSU(
      model,
      optimizer,
      amplification_scale=args.amplification_scale,
      perturbation_radius=args.perturbation_radius,
  )
  batches = training.PairedBatches(
      train_set,
      split,
      args.batch_size,
      seed=args.run_seed,
      num_workers=args.num_workers,
      device=args.device,
  )
  history = training.unlearn(
      method,
      batches,
      args.epochs,
      args.schedule,
      evaluate=_with_progress(
          _evaluator(train_set, test_set, split, args.num_workers)
      ),
  )
  fingerprints = {
      'checkpoint': reproducibility.fingerprint(checkpoint),
      **_split_fingerprints(split),
  }
  if args.save_model is not None:
    _save_model(args.save_model, model)
    fingerprints['model'] = reproducibility.fingerprint(args.save_model)
  _write(
      args.output,
      {
          'command': 'unlearn',
          'settings': {
              'learning_rate': args.learning_rate,
              'amplification_scale': args.amplification_scale,
              'perturbation_radius': args.perturbation_radius,
              'momentum': args.momentum,
              'weight_decay': args.weight_decay,
              'batch_size': args.batch_size,
              'epochs': args.epochs,
              'schedule': args.schedule,
          },
          'forget': _forget(args),
          'sizes': _sizes(split),
          'run_seed': args.run_seed,
          'epochs': history,
          'fingerprints': fingerprints,
          'environment': environment,
      },
  )
  print(json.dumps(history[-1]['metrics'], indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
  checkpoint = args.checkpoint or _hugging_face_model(
      args.parser, f'retrain_ratio_{args.ratio}_seed_{args.seed}.pt'
  )
  _check_inputs(args.parser, checkpoint)
  _prepare_outputs(args.parser, args.output)
  environment = reproducibility.environment(args.device)
  reproducibility.make_deterministic()
  train_set = cifar100.load(args.data_root, train=True)
  test_set = cifar100.load(args.data_root, train=False)
  split = _split(args, train_set, test_set)
  model = _load_model(checkpoint, len(train_set.classes), args.device)
  result = metrics.evaluate(
      model, train_set, test_set, split, num_workers=args.num_workers
  )
  _write(
      args.output,
      {
          'command': 'evaluate',
          'forget': _forget(args),
          'sizes': _sizes(split),
          'metrics': result,
          'fingerprints': {
              'checkpoint': reproducibility.fingerprint(checkpoint),
              **_split_fingerprints(split),
          },
          'environment': environment,
      },
  )
  print(json.dumps(result, indent=2, sort_keys=True))


def _check_inputs(
    parser: argparse.ArgumentParser, *paths: pathlib.Path
) -> None:
  for path in paths:
    if not path.is_file():
      parser.error(f'{path} does not exist.')


def _prepare_outputs(
    parser: argparse.ArgumentParser, *paths: pathlib.Path | None
) -> None:
  for path in paths:
    if path is None:
      continue
    if path.exists():
      parser.error(f'{path} already exists.')
    path.parent.mkdir(parents=True, exist_ok=True)


def _split(
    args: argparse.Namespace, train_set: data.Dataset, test_set: data.Dataset
) -> splits.Split:
  return splits.random_forgetting(
      train_set.targets, test_set.targets, args.ratio, args.seed
  )


def _hugging_face_model(
    parser: argparse.ArgumentParser, name: str
) -> pathlib.Path:
  if name not in _SHARED_MODELS:
    parser.error(f'{_HUGGING_FACE} has no {name}; give --checkpoint.')
  path = pathlib.Path('models', 'cifar100_resnet18', name)
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(
        f'{_HUGGING_FACE}/resolve/main/cifar100_resnet18/{name}',
        str(path),
        hash_prefix=_SHARED_MODELS[name],
    )
  return path


def _load_model(
    checkpoint: pathlib.Path, num_classes: int, device: str
) -> nn.Module:
  model = resnet18.ResNet18(num_classes=num_classes)
  model.load_state_dict(
      torch.load(checkpoint, map_location='cpu', weights_only=True)
  )
  return model.to(device)


def _evaluator(
    train_set: data.Dataset,
    test_set: data.Dataset,
    split: splits.Split,
    num_workers: int,
) -> training.Evaluate:
  return lambda model: metrics.evaluate(
      model, train_set, test_set, split, num_workers=num_workers
  )


def _test_accuracy(
    test_set: data.Dataset, indices: np.ndarray, num_workers: int
) -> training.Evaluate:
  return lambda model: {
      'test_accuracy': metrics.accuracy(
          metrics.outputs(model, test_set, indices, num_workers=num_workers)
      )
  }


def _with_progress(evaluate: training.Evaluate) -> training.Evaluate:
  epochs = itertools.count(1)

  def evaluate_and_print(model: nn.Module) -> dict[str, float]:
    result = evaluate(model)
    line = {'epoch': next(epochs), **result}
    print(json.dumps(line, sort_keys=True), file=sys.stderr, flush=True)
    return result

  return evaluate_and_print


def _forget(args: argparse.Namespace) -> dict[str, Any] | None:
  if args.ratio is None:
    return None
  return {'ratio': args.ratio, 'seed': args.seed}


def _sizes(split: splits.Split) -> dict[str, int]:
  return {name: len(indices) for name, indices in split._asdict().items()}


def _split_fingerprints(split: splits.Split) -> dict[str, str]:
  return {
      name: reproducibility.fingerprint(indices)
      for name, indices in split._asdict().items()
  }


def _save_model(path: pathlib.Path, model: nn.Module) -> None:
  with path.open('xb') as f:
    torch.save(model.state_dict(), f)


def _write(path: pathlib.Path, record: dict[str, Any]) -> None:
  with path.open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write('\n')
