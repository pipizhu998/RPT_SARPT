"""EATA test-time adaptation.

Adapted from the authors' EATA reference implementation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import islice
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from core.utils import progress_total


BatchNorm = nn.modules.batchnorm._BatchNorm
FisherState = dict[str, tuple[torch.Tensor, torch.Tensor]]


class EATA(nn.Module):
    """Efficient entropy-aware test-time adaptation.

    EATA updates BatchNorm affine parameters only on reliable and
    non-redundant samples, with an optional Fisher regularizer to reduce
    forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        fishers: FisherState | None = None,
        fisher_alpha: float = 2000.0,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float = math.log(10) * 0.4,
        d_margin: float = 0.05,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("EATA requires at least one adaptation step.")
        if fisher_alpha < 0.0:
            raise ValueError("EATA fisher_alpha must be non-negative.")
        if e_margin < 0.0:
            raise ValueError("EATA e_margin must be non-negative.")
        if d_margin < 0.0:
            raise ValueError("EATA d_margin must be non-negative.")

        self.model = model
        self.optimizer = optimizer
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.steps = steps
        self.episodic = episodic
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.current_model_probs: torch.Tensor | None = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model,
            self.optimizer,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
            self.current_model_probs = None

        outputs = None
        for _step in range(self.steps):
            outputs, selected_count, reliable_count, updated_probs = forward_and_adapt_eata(
                inputs=inputs,
                model=self.model,
                optimizer=self.optimizer,
                fishers=self.fishers,
                e_margin=self.e_margin,
                current_model_probs=self.current_model_probs,
                fisher_alpha=self.fisher_alpha,
                d_margin=self.d_margin,
            )
            self.num_samples_update_2 += selected_count
            self.num_samples_update_1 += reliable_count
            self.current_model_probs = updated_probs

        if outputs is None:
            raise RuntimeError("EATA did not produce outputs.")
        return outputs

    def reset(self) -> None:
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset EATA without saved model/optimizer state.")
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


def update_model_probs(
    current_model_probs: torch.Tensor | None,
    new_probs: torch.Tensor,
) -> torch.Tensor | None:
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        return new_probs.detach().mean(0)
    if new_probs.size(0) == 0:
        return current_model_probs.detach()
    return 0.9 * current_model_probs.detach() + 0.1 * new_probs.detach().mean(0)


@torch.enable_grad()
def forward_and_adapt_eata(
    inputs: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    fishers: FisherState | None,
    e_margin: float,
    current_model_probs: torch.Tensor | None,
    fisher_alpha: float = 2000.0,
    d_margin: float = 0.05,
) -> tuple[torch.Tensor, int, int, torch.Tensor | None]:
    """Forward one batch and adapt with EATA's sample filtering."""
    outputs = model(inputs)
    entropies = softmax_entropy(outputs)

    reliable_mask = entropies < e_margin
    reliable_indices = torch.where(reliable_mask)[0]
    reliable_count = int(reliable_indices.numel())
    reliable_probs = outputs[reliable_indices].softmax(1)

    if current_model_probs is not None and reliable_count > 0:
        cosine_similarities = F.cosine_similarity(
            current_model_probs.unsqueeze(0),
            reliable_probs,
            dim=1,
        )
        selected_mask = torch.abs(cosine_similarities) < d_margin
        selected_indices = reliable_indices[selected_mask]
        selected_probs = reliable_probs[selected_mask]
    else:
        selected_indices = reliable_indices
        selected_probs = reliable_probs

    updated_probs = update_model_probs(current_model_probs, selected_probs)
    selected_count = int(selected_indices.numel())
    if selected_count == 0:
        optimizer.zero_grad(set_to_none=True)
        return outputs, selected_count, reliable_count, updated_probs

    selected_entropies = entropies[selected_indices]
    entropy_weights = torch.exp(e_margin - selected_entropies.detach())
    loss = selected_entropies.mul(entropy_weights).mean(0)

    if fishers is not None:
        fisher_loss = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
        for name, parameter in model.named_parameters():
            if name in fishers:
                fisher, anchor = fishers[name]
                fisher_loss = fisher_loss + (fisher * (parameter - anchor).pow(2)).sum()
        loss = loss + fisher_alpha * fisher_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Evaluate the adapted model on the same batch, which is especially
    # important for episodic EATA where the next batch starts from a reset.
    with torch.no_grad():
        adapted_outputs = model(inputs)

    return adapted_outputs, selected_count, reliable_count, updated_probs


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
    """Configure a model for EATA adaptation."""
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
    """Validate that a model is configured for EATA."""
    if not model.training:
        raise AssertionError("EATA needs train mode: call model.train().")
    parameter_grads = [parameter.requires_grad for parameter in model.parameters()]
    if not any(parameter_grads):
        raise AssertionError("EATA needs parameters to update.")
    if all(parameter_grads):
        raise AssertionError("EATA should not update all model parameters.")
    if not any(isinstance(module, BatchNorm) for module in model.modules()):
        raise AssertionError("EATA needs BatchNorm layers for its optimization.")


def setup_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str = "sgd",
    lr: float = 2.5e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Set up the EATA optimizer."""
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if normalized_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported EATA optimizer: {optimizer_name}")


def compute_fishers(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_examples: int,
    max_batches: int | None = None,
    clip_by_norm: float | None = None,
) -> FisherState:
    """Estimate diagonal Fisher information from pseudo-labels."""
    if max_examples <= 0:
        return {}

    criterion = nn.CrossEntropyLoss()
    fishers: dict[str, torch.Tensor] = {}
    batches = 0
    examples = 0
    iterator = islice(dataloader, max_batches) if max_batches is not None else dataloader

    for inputs, _targets in iterator:
        if examples >= max_examples:
            break
        remaining = max_examples - examples
        if inputs.size(0) > remaining:
            inputs = inputs[:remaining]

        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        pseudo_targets = outputs.detach().argmax(dim=1)
        loss = criterion(outputs, pseudo_targets)

        model.zero_grad(set_to_none=True)
        loss.backward()
        batches += 1
        examples += inputs.size(0)

        for name, parameter in model.named_parameters():
            if parameter.grad is None or not parameter.requires_grad:
                continue
            fisher = parameter.grad.detach().pow(2)
            if clip_by_norm is not None:
                fisher = fisher.clamp(max=clip_by_norm)
            fishers[name] = fishers.get(name, torch.zeros_like(fisher)) + fisher

    model.zero_grad(set_to_none=True)
    if batches == 0:
        raise RuntimeError("No examples were available for EATA Fisher estimation.")

    return {
        name: (fisher / batches, dict(model.named_parameters())[name].detach().clone())
        for name, fisher in fishers.items()
    }


def setup_eata(
    model: nn.Module,
    optimizer_name: str = "sgd",
    lr: float = 2.5e-4,
    steps: int = 1,
    weight_decay: float = 0.0,
    episodic: bool = False,
    e_margin: float = math.log(10) * 0.4,
    d_margin: float = 0.05,
    fisher_dataloader: torch.utils.data.DataLoader | None = None,
    fisher_size: int = 0,
    fisher_alpha: float = 2000.0,
    fisher_clip_by_norm: float | None = None,
    device: torch.device | None = None,
) -> tuple[EATA, list[str]]:
    """Configure, optimize, and wrap a model with EATA."""
    model = configure_model(model)
    params, param_names = collect_params(model)
    if not params:
        raise RuntimeError("EATA requires affine BatchNorm parameters to adapt.")

    fishers = None
    if fisher_dataloader is not None and fisher_size > 0:
        fisher_device = device or next(model.parameters()).device
        fishers = compute_fishers(
            model=model,
            dataloader=fisher_dataloader,
            device=fisher_device,
            max_examples=fisher_size,
            clip_by_norm=fisher_clip_by_norm,
        )

    optimizer = setup_optimizer(
        params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    eata_model = EATA(
        model,
        optimizer,
        fishers=fishers,
        fisher_alpha=fisher_alpha,
        steps=steps,
        episodic=episodic,
        e_margin=e_margin,
        d_margin=d_margin,
    )
    check_model(eata_model.model)
    return eata_model, param_names
