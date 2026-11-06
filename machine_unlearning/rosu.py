import functools
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional


class ROSU:

  def __init__(
      self,
      model: nn.Module,
      optimizer: torch.optim.Optimizer,
      amplification_scale: float,
      perturbation_radius: float = 1e-3,
      stabilizer: float = 1e-8,
      degeneracy_threshold: float = 1e-6,
  ):
    _check_nonnegative('amplification_scale', amplification_scale)
    _check_positive('perturbation_radius', perturbation_radius)
    _check_positive('stabilizer', stabilizer)
    _check_positive('degeneracy_threshold', degeneracy_threshold)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    ]
    if not parameters:
      raise ValueError('The optimizer holds no parameter.')
    if not all(parameter.requires_grad for parameter in parameters):
      raise ValueError('All optimizer parameters must require gradients.')
    self.model = model
    self.optimizer = optimizer
    self.amplification_scale = amplification_scale
    self.perturbation_radius = perturbation_radius
    self.stabilizer = stabilizer
    self.degeneracy_threshold = degeneracy_threshold
    self._parameters = parameters
    self._sizes = [parameter.numel() for parameter in parameters]
    self._dtype = functools.reduce(
        torch.promote_types,
        [parameter.dtype for parameter in parameters],
        torch.float32,
    )

  def step(self, forget_batch: Any, retain_batch: Any) -> None:
    amplification_coefficient = (
        self.amplification_scale
        * self._learning_rate()
        / self.perturbation_radius
    )
    forget_gradient = self._gradient(forget_batch)
    retain_gradient = self._gradient(retain_batch)
    regularized_retain_norm_squared = (
        retain_gradient.dot(retain_gradient) + self.stabilizer
    )
    projected_forget_gradient = (
        forget_gradient
        - (
            forget_gradient.dot(retain_gradient)
            / regularized_retain_norm_squared
        )
        * retain_gradient
    )
    projected_norm = projected_forget_gradient.norm()
    if projected_norm.item() <= self.degeneracy_threshold:
      self._apply_optimizer(retain_gradient)
      return
    perturbation_direction = projected_forget_gradient / projected_norm
    perturbation = self.perturbation_radius * perturbation_direction
    jacobian_scale = self.perturbation_radius / projected_norm
    surrogate_retain_gradient = self._surrogate_gradient(
        perturbation, retain_batch
    )
    retain_component = (
        retain_gradient.dot(surrogate_retain_gradient)
        / regularized_retain_norm_squared
    ) * retain_gradient
    perturbation_component = (
        perturbation_direction.dot(surrogate_retain_gradient)
        * perturbation_direction
    )
    retain_correction = surrogate_retain_gradient + jacobian_scale * (
        surrogate_retain_gradient - retain_component - perturbation_component
    )
    self._add_to_parameters(amplification_coefficient * perturbation)
    self._apply_optimizer(retain_correction)

  def _learning_rate(self) -> float:
    learning_rates = {
        float(group['lr']) for group in self.optimizer.param_groups
    }
    if len(learning_rates) != 1:
      raise ValueError(
          f'Parameter groups must share one learning rate: {learning_rates}.'
      )
    learning_rate = learning_rates.pop()
    _check_positive('learning rate', learning_rate)
    return learning_rate

  def _gradient(self, batch: Any) -> torch.Tensor:
    inputs, labels = batch
    loss = functional.cross_entropy(self.model(inputs), labels)
    gradients = torch.autograd.grad(
        loss,
        self._parameters,
        allow_unused=True,
        materialize_grads=True,
    )
    return torch.cat(
        [gradient.reshape(-1).to(self._dtype) for gradient in gradients]
    )

  def _surrogate_gradient(
      self, perturbation: torch.Tensor, retain_batch: Any
  ) -> torch.Tensor:
    current_weights = [
        parameter.detach().clone() for parameter in self._parameters
    ]
    self._add_to_parameters(perturbation)
    try:
      return self._gradient(retain_batch)
    finally:
      with torch.no_grad():
        for parameter, weight in zip(self._parameters, current_weights):
          parameter.copy_(weight)

  @torch.no_grad()
  def _add_to_parameters(self, vector: torch.Tensor) -> None:
    for parameter, chunk in zip(self._parameters, self._split(vector)):
      parameter.add_(chunk.view_as(parameter).to(parameter.dtype))

  @torch.no_grad()
  def _apply_optimizer(self, gradient: torch.Tensor) -> None:
    for parameter, chunk in zip(self._parameters, self._split(gradient)):
      if parameter.grad is None:
        parameter.grad = torch.empty_like(parameter)
      parameter.grad.copy_(chunk.view_as(parameter))
    self.optimizer.step()

  def _split(self, vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return torch.split(vector, self._sizes)


def _check_positive(name: str, value: float) -> None:
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f'{name} must be finite and positive, got {value}.')


def _check_nonnegative(name: str, value: float) -> None:
  if not math.isfinite(value) or value < 0:
    raise ValueError(f'{name} must be finite and nonnegative, got {value}.')
