from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms

from data_utils.data import DATASET_STATS, LoaderBundle, build_test_transform, build_torchvision_dataset, make_loader

from .augmentations import augment_and_mix


@dataclass(frozen=True)
class AugMixParams:
    severity: int = 3
    width: int = 3
    depth: int = -1
    alpha: float = 1.0
    all_ops: bool = True


class AugMixTrainTransform:
    """Return clean, AugMix-1, and AugMix-2 views for JSD training."""

    def __init__(self, params: AugMixParams, dataset_name: str = "cifar10") -> None:
        if dataset_name not in DATASET_STATS:
            raise ValueError(f"Unsupported AugMix dataset: {dataset_name}")
        self.params = params
        ops: list[object] = [transforms.RandomCrop(32, padding=4)]
        if dataset_name == "cifar10":
            ops.append(transforms.RandomHorizontalFlip())
        self.preprocess = transforms.Compose(ops)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(*DATASET_STATS[dataset_name])

    def _finish(self, image: Image.Image) -> torch.Tensor:
        return self.normalize(self.to_tensor(image))

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.preprocess(image.convert("RGB"))
        clean = self._finish(base)
        augmix_1 = self._finish(
            augment_and_mix(
                base,
                severity=self.params.severity,
                width=self.params.width,
                depth=self.params.depth,
                alpha=self.params.alpha,
                all_ops=self.params.all_ops,
            )
        )
        augmix_2 = self._finish(
            augment_and_mix(
                base,
                severity=self.params.severity,
                width=self.params.width,
                depth=self.params.depth,
                alpha=self.params.alpha,
                all_ops=self.params.all_ops,
            )
        )
        return clean, augmix_1, augmix_2


def build_augmix_train_transform(
    dataset_name: str = "cifar10",
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    all_ops: bool = True,
) -> AugMixTrainTransform:
    return AugMixTrainTransform(
        AugMixParams(
            severity=severity,
            width=width,
            depth=depth,
            alpha=alpha,
            all_ops=all_ops,
        ),
        dataset_name=dataset_name,
    )


def build_augmix_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    all_ops: bool = True,
    download: bool = False,
) -> LoaderBundle:
    return build_augmix_loaders(
        dataset_name="cifar10",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        seed=seed,
        severity=severity,
        width=width,
        depth=depth,
        alpha=alpha,
        all_ops=all_ops,
        download=download,
    )


def build_augmix_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    all_ops: bool = True,
    download: bool = False,
) -> LoaderBundle:
    train_transform = build_augmix_train_transform(
        dataset_name=dataset_name,
        severity=severity,
        width=width,
        depth=depth,
        alpha=alpha,
        all_ops=all_ops,
    )
    clean_transform = build_test_transform(dataset_name=dataset_name)

    train_dataset = build_torchvision_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        transform=train_transform,
        download=download,
    )
    val_dataset = build_torchvision_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        transform=clean_transform,
        download=download,
    )
    test_dataset = build_torchvision_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=False,
        transform=clean_transform,
        download=download,
    )

    total_items = len(train_dataset)
    val_items = int(total_items * val_ratio)
    train_items = total_items - val_items
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_items, generator=generator).tolist()
    train_indices = indices[:train_items]
    val_indices = indices[train_items:]

    return LoaderBundle(
        train=make_loader(Subset(train_dataset, train_indices), batch_size, num_workers, True),
        val=make_loader(Subset(val_dataset, val_indices), batch_size, num_workers, False),
        test=make_loader(test_dataset, batch_size, num_workers, False),
    )
