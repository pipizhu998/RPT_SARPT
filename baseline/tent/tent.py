from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
from torch import nn


BatchNorm = nn.modules.batchnorm._BatchNorm


class Tent(nn.Module):
    """Online test-time entropy minimization.

    Every forward pass updates BatchNorm affine parameters by minimizing
    prediction entropy, then returns logits from the just-adapted model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("TENT requires at least one adaptation step.")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model,
            self.optimizer,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()

        outputs = None
        for _step in range(self.steps):
            outputs = forward_and_adapt(inputs, self.model, self.optimizer)

        if outputs is None:
            raise RuntimeError("TENT did not produce outputs.")
        return outputs

    def reset(self) -> None:
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset TENT without saved model/optimizer state.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of the softmax distribution from logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(
    inputs: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Forward one batch, adapt by entropy, then return adapted logits."""
    outputs = model(inputs)
    loss = softmax_entropy(outputs).mean(0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        adapted_outputs = model(inputs)
    return adapted_outputs


def collect_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    """Collect BatchNorm affine scale and shift parameters."""
    params: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, BatchNorm):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name in {"weight", "bias"}:
                    params.append(parameter)
                    names.append(f"{module_name}.{parameter_name}")
    return params, names


def copy_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Copy model and optimizer states for episodic reset."""
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
) -> None:
    """Restore model and optimizer states."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
    """Configure a model for TENT adaptation."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, BatchNorm):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def check_model(model: nn.Module) -> None:
    """Validate that a model is configured for TENT."""
    if not model.training:
        raise AssertionError("TENT needs train mode: call model.train().")
    parameter_grads = [parameter.requires_grad for parameter in model.parameters()]
    if not any(parameter_grads):
        raise AssertionError("TENT needs parameters to update.")
    if all(parameter_grads):
        raise AssertionError("TENT should not update all model parameters.")
    if not any(isinstance(module, BatchNorm) for module in model.modules()):
        raise AssertionError("TENT needs BatchNorm layers for its optimization.")


def setup_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Set up the TENT optimizer."""
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if normalized_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported TENT optimizer: {optimizer_name}")


def setup_tent(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    steps: int = 1,
    weight_decay: float = 0.0,
    episodic: bool = False,
) -> tuple[Tent, list[str]]:
    """Configure, optimize, and wrap a model with TENT.

    This mutates the supplied model in place. Use evaluate_model_tent for the
    side-effect-safe evaluation entry point, or pass a deepcopy here directly.
    """
    model = configure_model(model)
    params, param_names = collect_params(model)
    if not params:
        raise RuntimeError("TENT requires affine BatchNorm parameters to adapt.")
    optimizer = setup_optimizer(
        params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    tent_model = Tent(model, optimizer, steps=steps, episodic=episodic)
    check_model(tent_model.model)
    return tent_model, param_names
