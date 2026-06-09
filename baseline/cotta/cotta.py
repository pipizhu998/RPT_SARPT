"""CoTTA continual test-time adaptation.

Adapted from the authors' CoTTA reference implementation while keeping the
evaluation interface consistent with the other baselines in this repository.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from torchvision.transforms import InterpolationMode

from data_utils.data import DATASET_STATS


BatchNorm = nn.modules.batchnorm._BatchNorm


class CoTTATransform:
    """Tensor-space test-time augmentations used by CoTTA.

    The original CoTTA transform operates on image tensors clipped to [0, 1].
    This adapter denormalizes local dataloader tensors before augmentation and
    normalizes them back afterward, so CoTTA sees the same input convention as
    the evaluated model.
    """

    def __init__(
        self,
        normalization_dataset: str,
        gaussian_std: float = 0.005,
        soft: bool = False,
        clip_inputs: bool = True,
    ) -> None:
        if normalization_dataset not in DATASET_STATS:
            raise ValueError(f"Unsupported normalization dataset: {normalization_dataset}")
        mean, std = DATASET_STATS[normalization_dataset]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.gaussian_std = gaussian_std
        self.soft = soft
        self.clip_inputs = clip_inputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=inputs.device, dtype=inputs.dtype)
        std = self.std.to(device=inputs.device, dtype=inputs.dtype)
        images = inputs.mul(std).add(mean)
        if self.clip_inputs:
            images = images.clamp(0.0, 1.0)

        images = self._color_jitter(images)
        images = TF.pad(images, padding=[16, 16], padding_mode="edge")
        images = TF.affine(
            images,
            angle=self._uniform(-8.0, 8.0) if self.soft else self._uniform(-15.0, 15.0),
            translate=[
                int(round(self._uniform(-2.0, 2.0))),
                int(round(self._uniform(-2.0, 2.0))),
            ],
            scale=self._uniform(0.95, 1.05) if self.soft else self._uniform(0.9, 1.1),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        images = TF.gaussian_blur(
            images,
            kernel_size=[5, 5],
            sigma=[
                0.001,
                self._uniform(0.001, 0.25) if self.soft else self._uniform(0.001, 0.5),
            ],
        )
        images = TF.center_crop(images, [32, 32])
        if torch.rand((), device=inputs.device) < 0.5:
            images = TF.hflip(images)
        images = images + torch.randn_like(images) * self.gaussian_std
        images = images.clamp(0.0, 1.0)
        return images.sub(mean).div(std)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return torch.empty(()).uniform_(low, high).item()

    def _color_jitter(self, images: torch.Tensor) -> torch.Tensor:
        brightness = (0.8, 1.2) if self.soft else (0.6, 1.4)
        contrast = (0.85, 1.15) if self.soft else (0.7, 1.3)
        saturation = (0.75, 1.25) if self.soft else (0.5, 1.5)
        hue = (-0.03, 0.03) if self.soft else (-0.06, 0.06)
        gamma = (0.85, 1.15) if self.soft else (0.7, 1.3)

        for fn_id in torch.randperm(5):
            if fn_id == 0:
                images = TF.adjust_brightness(images, self._uniform(*brightness))
            elif fn_id == 1:
                images = TF.adjust_contrast(images, self._uniform(*contrast))
            elif fn_id == 2:
                images = TF.adjust_saturation(images, self._uniform(*saturation))
            elif fn_id == 3:
                images = TF.adjust_hue(images, self._uniform(*hue))
            elif fn_id == 4:
                images = TF.adjust_gamma(images.clamp(1e-8, 1.0), self._uniform(*gamma))
        return images


class CoTTA(nn.Module):
    """Continual test-time adaptation with teacher EMA and stochastic restore."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        mt_alpha: float = 0.999,
        rst_m: float = 0.01,
        ap: float = 0.92,
        augmentation_views: int = 32,
        transform: CoTTATransform | None = None,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("CoTTA requires at least one adaptation step.")
        if not 0.0 <= mt_alpha <= 1.0:
            raise ValueError("CoTTA mt_alpha must be in [0, 1].")
        if not 0.0 <= rst_m <= 1.0:
            raise ValueError("CoTTA rst_m must be in [0, 1].")
        if augmentation_views < 0:
            raise ValueError("CoTTA augmentation_views must be non-negative.")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.mt_alpha = mt_alpha
        self.rst_m = rst_m
        self.ap = ap
        self.augmentation_views = augmentation_views
        self.transform = transform
        (
            self.model_state,
            self.optimizer_state,
            self.model_ema,
            self.model_anchor,
        ) = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()

        outputs = None
        for _step in range(self.steps):
            outputs = self.forward_and_adapt(inputs, self.model, self.optimizer)

        if outputs is None:
            raise RuntimeError("CoTTA did not produce outputs.")
        return outputs

    def reset(self) -> None:
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset CoTTA without saved model/optimizer state.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        (
            self.model_state,
            self.optimizer_state,
            self.model_ema,
            self.model_anchor,
        ) = copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()
    def forward_and_adapt(
        self,
        inputs: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        outputs = model(inputs)

        with torch.no_grad():
            anchor_prob = F.softmax(self.model_anchor(inputs), dim=1).max(1)[0]
            standard_ema = self.model_ema(inputs)
            outputs_ema = standard_ema
            if self.transform is not None and self.augmentation_views > 0:
                augmented_outputs = [
                    self.model_ema(self.transform(inputs)).detach()
                    for _ in range(self.augmentation_views)
                ]
                if anchor_prob.mean(0) < self.ap:
                    outputs_ema = torch.stack(augmented_outputs).mean(0)

        loss = softmax_entropy(outputs, outputs_ema.detach()).mean(0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        update_ema_variables(self.model_ema, self.model, self.mt_alpha)
        stochastic_restore(self.model, self.model_state, self.rst_m)
        return outputs_ema


def softmax_entropy(logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Cross entropy from teacher soft labels to student log probabilities."""
    return -(teacher_logits.softmax(1) * logits.log_softmax(1)).sum(1)


def update_ema_variables(
    ema_model: nn.Module,
    model: nn.Module,
    alpha_teacher: float,
) -> nn.Module:
    for ema_param, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(parameter.data, alpha=1.0 - alpha_teacher)
    return ema_model


def stochastic_restore(
    model: nn.Module,
    model_state: dict[str, torch.Tensor],
    restore_probability: float,
) -> None:
    if restore_probability <= 0.0:
        return
    with torch.no_grad():
        for module_name, module in model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name not in {"weight", "bias"} or not parameter.requires_grad:
                    continue
                state_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                anchor = model_state[state_name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                mask = torch.rand_like(parameter) < restore_probability
                parameter.data.copy_(torch.where(mask, anchor, parameter.data))


def collect_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    """Collect trainable weight and bias parameters, matching CoTTA."""
    params: list[nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name in {"weight", "bias"} and parameter.requires_grad:
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return params, names


def copy_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict, nn.Module, nn.Module]:
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    anchor_model = deepcopy(model)
    for parameter in ema_model.parameters():
        parameter.detach_()
    for parameter in anchor_model.parameters():
        parameter.detach_()
    return model_state, optimizer_state, ema_model, anchor_model


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
    """Configure a model for CoTTA adaptation.

    CoTTA's reference implementation adapts trainable weight/bias parameters
    broadly, while forcing BatchNorm layers to use current batch statistics.
    """
    model.train()
    model.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, BatchNorm):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def check_model(model: nn.Module) -> None:
    if not model.training:
        raise AssertionError("CoTTA needs train mode: call model.train().")
    parameter_grads = [parameter.requires_grad for parameter in model.parameters()]
    if not any(parameter_grads):
        raise AssertionError("CoTTA needs parameters to update.")
    if not any(isinstance(module, BatchNorm) for module in model.modules()):
        raise AssertionError("CoTTA needs BatchNorm layers for its optimization.")


def setup_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 0.9,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
    if normalized_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported CoTTA optimizer: {optimizer_name}")


def setup_cotta(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    steps: int = 1,
    weight_decay: float = 0.0,
    episodic: bool = False,
    mt_alpha: float = 0.999,
    rst_m: float = 0.01,
    ap: float = 0.92,
    augmentation_views: int = 32,
    normalization_dataset: str = "cifar10",
    gaussian_std: float = 0.005,
    soft_augmentations: bool = False,
    beta: float = 0.9,
) -> tuple[CoTTA, list[str]]:
    model = configure_model(model)
    params, param_names = collect_params(model)
    if not params:
        raise RuntimeError("CoTTA requires trainable weight/bias parameters to adapt.")
    optimizer = setup_optimizer(
        params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        beta=beta,
    )
    transform = CoTTATransform(
        normalization_dataset=normalization_dataset,
        gaussian_std=gaussian_std,
        soft=soft_augmentations,
    )
    cotta_model = CoTTA(
        model,
        optimizer,
        steps=steps,
        episodic=episodic,
        mt_alpha=mt_alpha,
        rst_m=rst_m,
        ap=ap,
        augmentation_views=augmentation_views,
        transform=transform,
    )
    check_model(cotta_model.model)
    return cotta_model, param_names
