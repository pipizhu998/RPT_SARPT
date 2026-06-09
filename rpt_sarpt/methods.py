from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as TF

from baseline.augmix.augmentations import augment_and_mix
from baseline.augmix.losses import stable_softmax
from data_utils.data import DATASET_STATS


BatchNorm = nn.modules.batchnorm._BatchNorm
BatchAugmenter = Callable[[torch.Tensor], torch.Tensor]
AdaptInputs = (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | list[torch.Tensor]
)


@dataclass(frozen=True)
class AugMixTTAParams:
    severity: int = 3
    width: int = 3
    depth: int = -1
    alpha: float = 1.0
    all_ops: bool = True


class AugMixBatchAugmenter:
    def __init__(
        self,
        mean: Iterable[float],
        std: Iterable[float],
        params: AugMixTTAParams | None = None,
    ) -> None:
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.params = params or AugMixTTAParams()
        if len(self.mean) != len(self.std):
            raise ValueError("AugMix TTA mean and std must have the same length.")
        if self.params.alpha <= 0.0:
            raise ValueError("AugMix alpha must be positive.")
        if self.params.width <= 0:
            raise ValueError("AugMix width must be positive.")

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        params: AugMixTTAParams | None = None,
    ) -> "AugMixBatchAugmenter":
        if dataset_name not in DATASET_STATS:
            raise ValueError(f"Unsupported AugMix TTA normalization dataset: {dataset_name}")
        mean, std = DATASET_STATS[dataset_name]
        return cls(mean=mean, std=std, params=params)

    def _stats_tensor(
        self,
        inputs: torch.Tensor,
        values: tuple[float, ...],
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("AugMix TTA expects image batches with shape [N, C, H, W].")
        if inputs.size(1) != len(values):
            raise ValueError(
                f"AugMix TTA stats have {len(values)} channels, "
                f"but inputs have {inputs.size(1)} channels."
            )
        return torch.as_tensor(values, device=inputs.device, dtype=inputs.dtype).view(
            1,
            -1,
            1,
            1,
        )

    def denormalize(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self._stats_tensor(inputs, self.mean)
        std = self._stats_tensor(inputs, self.std)
        return (inputs * std + mean).clamp(0.0, 1.0)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self._stats_tensor(inputs, self.mean)
        std = self._stats_tensor(inputs, self.std)
        return (inputs - mean) / std

    @torch.no_grad()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        dtype = inputs.dtype
        denormalized = self.denormalize(inputs).detach().float().cpu()
        augmented = []
        for image in denormalized:
            image_pil = TF.to_pil_image(image)
            mixed_pil = augment_and_mix(
                image_pil,
                severity=self.params.severity,
                width=self.params.width,
                depth=self.params.depth,
                alpha=self.params.alpha,
                all_ops=self.params.all_ops,
            )
            augmented.append(TF.to_tensor(mixed_pil))
        augmented_batch = torch.stack(augmented, dim=0).to(device=device, dtype=dtype)
        return self.normalize(augmented_batch)


class RPT(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        augmenter: BatchAugmenter | None = None,
        jsd_weight: float = 0.1,
        source_model: nn.Module | None = None,
        source_anchor_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("RPT requires at least one adaptation step.")
        if jsd_weight < 0.0:
            raise ValueError("JSD weight must be non-negative.")
        if jsd_weight > 0.0 and augmenter is None:
            raise ValueError("RPT requires an augmenter when JSD weight is positive.")
        if source_anchor_weight < 0.0:
            raise ValueError("Source anchor weight must be non-negative.")
        if source_anchor_weight > 0.0 and source_model is None:
            raise ValueError(
                "SARPT requires a source model when source anchor weight is positive."
            )
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.augmenter = augmenter
        self.jsd_weight = jsd_weight
        self.source_model = source_model
        self.source_anchor_weight = source_anchor_weight
        if self.source_model is not None:
            self.source_model.eval()
            self.source_model.requires_grad_(False)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model,
            self.optimizer,
        )

    def forward(self, inputs: AdaptInputs) -> torch.Tensor:
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(
                inputs=inputs,
                model=self.model,
                optimizer=self.optimizer,
                augmenter=self.augmenter,
                jsd_weight=self.jsd_weight,
                source_model=self.source_model,
                source_anchor_weight=self.source_anchor_weight,
            )

        if outputs is None:
            raise RuntimeError("RPT did not produce outputs.")
        return outputs

    def reset(self) -> None:
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset RPT without saved model/optimizer state.")
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def jsd_consistency_loss_per_sample(
    logits_clean: torch.Tensor,
    logits_augmix_1: torch.Tensor,
    logits_augmix_2: torch.Tensor,
) -> torch.Tensor:
    p_clean = stable_softmax(logits_clean)
    p_augmix_1 = stable_softmax(logits_augmix_1)
    p_augmix_2 = stable_softmax(logits_augmix_2)

    log_p_mixture = torch.clamp(
        (p_clean + p_augmix_1 + p_augmix_2) / 3.0,
        min=1e-7,
        max=1.0,
    ).log()

    return (
        F.kl_div(log_p_mixture, p_clean, reduction="none").sum(dim=1)
        + F.kl_div(log_p_mixture, p_augmix_1, reduction="none").sum(dim=1)
        + F.kl_div(log_p_mixture, p_augmix_2, reduction="none").sum(dim=1)
    ) / 3.0


def source_anchor_loss(
    logits_current: torch.Tensor,
    logits_source: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        source_probs = stable_softmax(logits_source)
    return F.kl_div(
        logits_current.log_softmax(dim=1),
        source_probs,
        reduction="batchmean",
    )


def split_adapt_inputs(
    inputs: AdaptInputs,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(inputs, torch.Tensor):
        return inputs, None, None
    if len(inputs) != 3:
        raise ValueError("RPT precomputed inputs must contain clean, augmix_1, and augmix_2.")
    clean, augmix_1, augmix_2 = inputs
    return clean, augmix_1, augmix_2


@torch.enable_grad()
def forward_and_adapt(
    inputs: AdaptInputs,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    augmenter: BatchAugmenter | None = None,
    jsd_weight: float = 0.1,
    source_model: nn.Module | None = None,
    source_anchor_weight: float = 0.0,
) -> torch.Tensor:
    if jsd_weight < 0.0:
        raise ValueError("JSD weight must be non-negative.")
    if source_anchor_weight < 0.0:
        raise ValueError("Source anchor weight must be non-negative.")
    if source_anchor_weight > 0.0 and source_model is None:
        raise ValueError(
            "SARPT requires a source model when source anchor weight is positive."
        )

    inputs_clean, inputs_augmix_1, inputs_augmix_2 = split_adapt_inputs(inputs)
    outputs = model(inputs_clean)
    entropy_per_sample = softmax_entropy(outputs)
    loss_ent = entropy_per_sample.mean(0)
    loss = loss_ent

    if jsd_weight > 0.0:
        if inputs_augmix_1 is None or inputs_augmix_2 is None:
            if augmenter is None:
                raise ValueError("RPT requires an augmenter when JSD weight is positive.")
            inputs_augmix_1 = augmenter(inputs_clean)
            inputs_augmix_2 = augmenter(inputs_clean)
        elif (
            inputs_augmix_1.shape != inputs_clean.shape
            or inputs_augmix_2.shape != inputs_clean.shape
        ):
            raise ValueError("Precomputed AugMix views must match the clean input shape.")
        logits_augmix_1 = model(inputs_augmix_1)
        logits_augmix_2 = model(inputs_augmix_2)
        jsd_per_sample = jsd_consistency_loss_per_sample(
            outputs,
            logits_augmix_1,
            logits_augmix_2,
        )
        loss_jsd = jsd_per_sample.mean(0)
        loss = loss + jsd_weight * loss_jsd

    if source_anchor_weight > 0.0:
        if source_model is None:
            raise RuntimeError("SARPT source model unexpectedly missing.")
        with torch.no_grad():
            source_logits = source_model(inputs_clean)
        loss_src = source_anchor_loss(
            logits_current=outputs,
            logits_source=source_logits,
        )
        loss = loss + source_anchor_weight * loss_src

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite RPT loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        adapted_outputs = model(inputs_clean)
    return adapted_outputs


def collect_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
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
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
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
    if not model.training:
        raise AssertionError("RPT needs train mode: call model.train().")
    parameter_grads = [parameter.requires_grad for parameter in model.parameters()]
    if not any(parameter_grads):
        raise AssertionError("RPT needs parameters to update.")
    if all(parameter_grads):
        raise AssertionError("RPT should not update all model parameters.")
    if not any(isinstance(module, BatchNorm) for module in model.modules()):
        raise AssertionError("RPT needs BatchNorm layers for its optimization.")


def setup_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    normalized_name = optimizer_name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if normalized_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported RPT optimizer: {optimizer_name}")


def setup_rpt(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    steps: int = 1,
    weight_decay: float = 0.0,
    episodic: bool = False,
    jsd_weight: float = 0.1,
    normalization_dataset: str | None = None,
    augmix_severity: int = 3,
    augmix_width: int = 3,
    augmix_depth: int = -1,
    augmix_alpha: float = 1.0,
    augmix_all_ops: bool = True,
    source_anchor_weight: float = 0.0,
) -> tuple[RPT, list[str]]:
    if source_anchor_weight < 0.0:
        raise ValueError("Source anchor weight must be non-negative.")

    source_model = None
    if source_anchor_weight > 0.0:
        source_model = deepcopy(model)
        source_model.eval()
        source_model.requires_grad_(False)

    augmenter = None
    if jsd_weight > 0.0:
        if normalization_dataset is None:
            raise ValueError("RPT requires normalization_dataset when JSD weight is positive.")
        augmenter = AugMixBatchAugmenter.from_dataset(
            normalization_dataset,
            params=AugMixTTAParams(
                severity=augmix_severity,
                width=augmix_width,
                depth=augmix_depth,
                alpha=augmix_alpha,
                all_ops=augmix_all_ops,
            ),
        )

    model = configure_model(model)
    params, param_names = collect_params(model)
    if not params:
        raise RuntimeError("RPT requires affine BatchNorm parameters to adapt.")
    optimizer = setup_optimizer(
        params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    rpt_model = RPT(
        model,
        optimizer,
        steps=steps,
        episodic=episodic,
        augmenter=augmenter,
        jsd_weight=jsd_weight,
        source_model=source_model,
        source_anchor_weight=source_anchor_weight,
    )
    check_model(rpt_model.model)
    return rpt_model, param_names
