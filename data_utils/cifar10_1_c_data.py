from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_utils.cifar10_1_data import CIFAR101Dataset
from data_utils.cifar10c_data import cifar10c_transform, sample_condition_indices


CIFAR10_1_C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
)


def cifar10_1_c_required_files(
    data_dir: str | Path,
    corruptions: list[str],
) -> list[Path]:
    root = Path(data_dir)
    required = [root / "labels.npy", root / "source_indices.npy", root / "metadata.json"]
    required.extend(root / f"{corruption}.npy" for corruption in corruptions)
    return required


def missing_cifar10_1_c_files(
    data_dir: str | Path,
    corruptions: list[str],
) -> list[Path]:
    return [
        path
        for path in cifar10_1_c_required_files(data_dir, corruptions)
        if not path.exists()
    ]


def ensure_cifar10_1_c_files(data_dir: str | Path, corruptions: list[str]) -> None:
    missing = missing_cifar10_1_c_files(data_dir, corruptions)
    if not missing:
        return
    missing_list = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "Missing CIFAR-10.1-C files:\n"
        f"{missing_list}\n\n"
        "Download the dataset first with:\n"
        "  bash tools/download_cifar10_1_c.sh"
    )


def _condition_size(images: np.ndarray, severity_count: int = 5) -> int:
    if len(images) % severity_count != 0:
        raise ValueError(
            f"Expected image count to be divisible by {severity_count}, got {len(images)}."
        )
    return len(images) // severity_count


class CIFAR101CDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        max_examples: int | None = None,
        seed: int = 0,
    ) -> None:
        if corruption not in CIFAR10_1_C_CORRUPTIONS:
            raise ValueError(f"Unsupported CIFAR-10.1-C corruption: {corruption}")
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("CIFAR-10.1-C severity must be one of 1, 2, 3, 4, 5.")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = cifar10c_transform()

        image_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10.1-C file: {image_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10.1-C labels file: {labels_path}")

        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.condition_size = _condition_size(self.images)
        self.start = (severity - 1) * self.condition_size
        self.end = severity * self.condition_size
        self.indices = self.start + sample_condition_indices(
            total_examples=self.condition_size,
            max_examples=max_examples,
            seed=seed,
        )

        if len(self.labels) not in {self.condition_size, len(self.images)}:
            raise ValueError(
                "CIFAR-10.1-C labels.npy must contain either one severity block "
                "or one label per image."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_index = int(self.indices[index])
        image = Image.fromarray(np.asarray(self.images[image_index], dtype=np.uint8))
        label_index = image_index if len(self.labels) == len(self.images) else image_index - self.start
        label = int(self.labels[label_index])
        return self.transform(image), label


class MixedCIFAR101CDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruptions: list[str],
        severity_levels: list[int],
        cifar10_1_data_dir: str = "datasets/CIFAR-10.1",
        cifar10_1_version: str = "v6",
        clean_download: bool = True,
        include_clean: bool = False,
        seed: int = 0,
        max_examples: int | None = None,
        max_examples_per_condition: int | None = None,
    ) -> None:
        if not corruptions and not include_clean:
            raise ValueError("Mixed CIFAR-10.1-C requires at least one source.")
        invalid_corruptions = sorted(set(corruptions) - set(CIFAR10_1_C_CORRUPTIONS))
        if invalid_corruptions:
            raise ValueError(f"Unsupported CIFAR-10.1-C corruptions: {invalid_corruptions}")
        invalid_severities = sorted(set(severity_levels) - {1, 2, 3, 4, 5})
        if invalid_severities:
            raise ValueError(
                "CIFAR-10.1-C severity must be one of 1, 2, 3, 4, 5; "
                f"got {invalid_severities}."
            )

        self.root = Path(root)
        self.corruptions = list(corruptions)
        self.severity_levels = list(severity_levels)
        self.transform = cifar10c_transform()

        labels_path = self.root / "labels.npy"
        source_indices_path = self.root / "source_indices.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10.1-C labels file: {labels_path}")
        if include_clean and not source_indices_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10.1-C source indices file: {source_indices_path}"
            )

        self.labels = np.load(labels_path, mmap_mode="r")
        self.images_by_corruption = {}
        condition_size: int | None = None
        for corruption in self.corruptions:
            image_path = self.root / f"{corruption}.npy"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10.1-C file: {image_path}")
            images = np.load(image_path, mmap_mode="r")
            current_size = _condition_size(images)
            if condition_size is None:
                condition_size = current_size
            elif condition_size != current_size:
                raise ValueError("All CIFAR-10.1-C corruption files must have the same size.")
            self.images_by_corruption[corruption] = images

        if condition_size is None:
            if len(self.labels) % 5 != 0:
                raise ValueError("Cannot infer CIFAR-10.1-C condition size from labels.")
            condition_size = len(self.labels) // 5
        self.condition_size = condition_size
        if len(self.labels) not in {self.condition_size, self.condition_size * 5}:
            raise ValueError(
                "CIFAR-10.1-C labels.npy must contain either one severity block "
                "or one label per image."
            )

        local_indices = sample_condition_indices(
            total_examples=self.condition_size,
            max_examples=max_examples_per_condition,
            seed=seed,
        )
        samples: list[tuple[int, int, int]] = []
        if include_clean:
            samples.extend((-1, int(local_index), int(local_index)) for local_index in local_indices)
        for corruption_index, _corruption in enumerate(self.corruptions):
            for severity in self.severity_levels:
                offset = (severity - 1) * self.condition_size
                samples.extend(
                    (
                        corruption_index,
                        int(offset + local_index),
                        int(local_index),
                    )
                    for local_index in local_indices
                )
        self.samples = samples

        order = np.arange(len(samples), dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("mixed_max_examples must be positive when provided.")
            order = order[:max_examples]
        self.order = order

        self.clean_dataset = None
        self.source_indices = None
        if include_clean:
            self.source_indices = np.load(source_indices_path, mmap_mode="r")
            if len(self.source_indices) < self.condition_size:
                raise ValueError(
                    "source_indices.npy must have at least one entry per clean condition image."
                )
            self.clean_dataset = CIFAR101Dataset(
                root=cifar10_1_data_dir,
                version=cifar10_1_version,
                download=clean_download,
                normalization_dataset="cifar10",
            )

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample_index = int(self.order[index])
        corruption_index, image_index, local_index = self.samples[sample_index]
        if corruption_index < 0:
            if self.clean_dataset is None or self.source_indices is None:
                raise RuntimeError("Clean CIFAR-10.1 dataset was not initialized.")
            return self.clean_dataset[int(self.source_indices[local_index])]

        corruption = self.corruptions[corruption_index]
        image = Image.fromarray(
            np.asarray(self.images_by_corruption[corruption][image_index], dtype=np.uint8)
        )
        label_index = image_index if len(self.labels) == self.condition_size * 5 else local_index
        label = int(self.labels[label_index])
        return self.transform(image), label


def build_cifar10_1_c_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruption: str,
    severity: int,
    max_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    dataset = CIFAR101CDataset(
        root=data_dir,
        corruption=corruption,
        severity=severity,
        max_examples=max_examples,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_mixed_cifar10_1_c_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruptions: list[str],
    severity_levels: list[int],
    cifar10_1_data_dir: str = "datasets/CIFAR-10.1",
    cifar10_1_version: str = "v6",
    clean_download: bool = True,
    include_clean: bool = False,
    seed: int = 0,
    max_examples: int | None = None,
    max_examples_per_condition: int | None = None,
) -> DataLoader:
    dataset = MixedCIFAR101CDataset(
        root=data_dir,
        corruptions=corruptions,
        severity_levels=severity_levels,
        cifar10_1_data_dir=cifar10_1_data_dir,
        cifar10_1_version=cifar10_1_version,
        clean_download=clean_download,
        include_clean=include_clean,
        seed=seed,
        max_examples=max_examples,
        max_examples_per_condition=max_examples_per_condition,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
