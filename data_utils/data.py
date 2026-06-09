from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
NOISE_TYPES = ("none", "gaussian", "speckle", "salt_pepper", "mixed")


DATASET_STATS = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "svhn": (SVHN_MEAN, SVHN_STD),
    "mnist": (MNIST_MEAN, MNIST_STD),
}


def apply_noise(image: torch.Tensor, noise_type: str, severity: float) -> torch.Tensor:
    if noise_type == "none" or severity <= 0.0:
        return image

    image = image.clone()

    if noise_type == "gaussian":
        image = image + torch.randn_like(image) * severity
    elif noise_type == "speckle":
        image = image + image * torch.randn_like(image) * severity
    elif noise_type == "salt_pepper":
        mask = torch.rand((1, image.shape[1], image.shape[2]), dtype=image.dtype)
        salt_mask = mask < (severity / 2.0)
        pepper_mask = (mask >= (severity / 2.0)) & (mask < severity)
        image = image.masked_fill(salt_mask.expand_as(image), 1.0)
        image = image.masked_fill(pepper_mask.expand_as(image), 0.0)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return image.clamp_(0.0, 1.0)


class RandomNoiseAugment:
    def __init__(
        self,
        noise_type: str,
        p: float,
        min_severity: float,
        max_severity: float,
    ) -> None:
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_type}")
        self.noise_type = noise_type
        self.p = p
        self.min_severity = min_severity
        self.max_severity = max_severity

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "none" or self.p <= 0.0:
            return image
        if random.random() > self.p:
            return image

        severity = random.uniform(self.min_severity, self.max_severity)
        noise_type = self.noise_type
        if noise_type == "mixed":
            noise_type = random.choice(["gaussian", "speckle", "salt_pepper"])
        return apply_noise(image, noise_type, severity)


@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_train_transform(
    dataset_name: str = "cifar10",
    train_noise_type: str = "none",
    train_noise_prob: float = 0.0,
    train_noise_min: float = 0.0,
    train_noise_max: float = 0.0,
) -> transforms.Compose:
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unsupported train transform dataset: {dataset_name}")

    ops: list[object] = [transforms.RandomCrop(32, padding=4)]
    if dataset_name == "cifar10":
        ops.append(transforms.RandomHorizontalFlip())
    if dataset_name == "mnist":
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())
    if train_noise_type != "none" and train_noise_prob > 0.0:
        ops.append(
            RandomNoiseAugment(
                noise_type=train_noise_type,
                p=train_noise_prob,
                min_severity=train_noise_min,
                max_severity=train_noise_max,
            )
        )
    mean, std = DATASET_STATS[dataset_name]
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def build_test_transform(
    dataset_name: str = "cifar10",
    normalization_dataset: str | None = None,
) -> transforms.Compose:
    normalization_dataset = normalization_dataset or dataset_name
    if normalization_dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported normalization dataset: {normalization_dataset}")

    ops: list[object] = []
    if dataset_name in {"mnist", "mnistc"}:
        ops.extend([transforms.Resize(32), transforms.Grayscale(num_output_channels=3)])
    ops.append(transforms.ToTensor())
    mean, std = DATASET_STATS[normalization_dataset]
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def build_torchvision_dataset(
    dataset_name: str,
    data_dir: str,
    train: bool,
    transform: Callable | None,
    download: bool,
) -> torch.utils.data.Dataset:
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    if dataset_name == "svhn":
        return datasets.SVHN(
            root=data_dir,
            split="train" if train else "test",
            transform=transform,
            download=download,
        )
    if dataset_name == "mnist":
        return datasets.MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    raise ValueError(f"Unsupported torchvision dataset: {dataset_name}")


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    train_noise_type: str = "none",
    train_noise_prob: float = 0.0,
    train_noise_min: float = 0.0,
    train_noise_max: float = 0.0,
    download: bool = True,
) -> LoaderBundle:
    return build_classification_loaders(
        dataset_name="cifar10",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        seed=seed,
        train_noise_type=train_noise_type,
        train_noise_prob=train_noise_prob,
        train_noise_min=train_noise_min,
        train_noise_max=train_noise_max,
        download=download,
    )


def build_classification_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    train_noise_type: str = "none",
    train_noise_prob: float = 0.0,
    train_noise_min: float = 0.0,
    train_noise_max: float = 0.0,
    download: bool = True,
) -> LoaderBundle:
    train_transform = build_train_transform(
        dataset_name=dataset_name,
        train_noise_type=train_noise_type,
        train_noise_prob=train_noise_prob,
        train_noise_min=train_noise_min,
        train_noise_max=train_noise_max,
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

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return LoaderBundle(
        train=make_loader(train_subset, batch_size, num_workers, shuffle=True),
        val=make_loader(val_subset, batch_size, num_workers, shuffle=False),
        test=make_loader(test_dataset, batch_size, num_workers, shuffle=False),
    )


def build_cifar10_clean_test_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    download: bool = False,
) -> DataLoader:
    return build_clean_test_loader(
        dataset_name="cifar10",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )


def build_clean_test_loader(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    download: bool = False,
    normalization_dataset: str | None = None,
) -> DataLoader:
    dataset = build_torchvision_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=False,
        transform=build_test_transform(
            dataset_name=dataset_name,
            normalization_dataset=normalization_dataset,
        ),
        download=download,
    )
    return make_loader(dataset, batch_size, num_workers, shuffle=False)
