from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_utils.data import build_test_transform


DIGITROBUST_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "motion_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "rotate",
    "translate",
)


class DigitRobustArrayDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        subset: str,
        split: str = "test",
        corruption: str | None = None,
        normalization_dataset: str = "svhn",
    ) -> None:
        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.corruption = corruption

        if corruption is None:
            data_root = self.root / "clean" / subset / split
            transform_dataset = subset
        else:
            data_root = self.root / "corrupt" / f"{subset}-corrupt" / corruption
            transform_dataset = f"{subset}c"

        image_path = data_root / "images.npy"
        labels_path = data_root / "labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing DigitRobust images file: {image_path}.")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing DigitRobust labels file: {labels_path}.")

        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.transform = build_test_transform(
            dataset_name=transform_dataset,
            normalization_dataset=normalization_dataset,
        )

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"DigitRobust images/labels length mismatch in {data_root}: "
                f"{len(self.images)} != {len(self.labels)}."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(np.asarray(self.images[index], dtype=np.uint8), mode="RGB")
        label = int(self.labels[index])
        return self.transform(image), label


class MixedDigitRobustDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        subset: str | list[str],
        corruptions: list[str],
        include_clean: bool = True,
        seed: int = 0,
        max_examples: int | None = None,
        normalization_dataset: str = "svhn",
    ) -> None:
        if not corruptions and not include_clean:
            raise ValueError("Mixed DigitRobust requires at least one source dataset.")

        subsets = [subset] if isinstance(subset, str) else list(subset)
        if not subsets:
            raise ValueError("Mixed DigitRobust requires at least one subset.")

        self.datasets: list[Dataset] = []
        for subset_name in subsets:
            if include_clean:
                self.datasets.append(
                    DigitRobustArrayDataset(
                        root=root,
                        subset=subset_name,
                        split="test",
                        normalization_dataset=normalization_dataset,
                    )
                )
            self.datasets.extend(
                DigitRobustArrayDataset(
                    root=root,
                    subset=subset_name,
                    corruption=corruption,
                    normalization_dataset=normalization_dataset,
                )
                for corruption in corruptions
            )

        total_examples = sum(len(dataset) for dataset in self.datasets)
        order = np.arange(total_examples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be positive when provided.")
            order = order[:max_examples]
        self.order = order
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in self.datasets])

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        mixed_index = int(self.order[index])
        dataset_index = int(np.searchsorted(self.offsets, mixed_index, side="right") - 1)
        local_index = mixed_index - int(self.offsets[dataset_index])
        return self.datasets[dataset_index][local_index]


def make_digitrobust_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_digitrobust_clean_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    subset: str,
    split: str = "test",
    normalization_dataset: str = "svhn",
) -> DataLoader:
    return make_digitrobust_loader(
        DigitRobustArrayDataset(
            root=data_dir,
            subset=subset,
            split=split,
            normalization_dataset=normalization_dataset,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_digitrobust_corrupt_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    subset: str,
    corruption: str,
    normalization_dataset: str = "svhn",
) -> DataLoader:
    return make_digitrobust_loader(
        DigitRobustArrayDataset(
            root=data_dir,
            subset=subset,
            corruption=corruption,
            normalization_dataset=normalization_dataset,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_mixed_digitrobust_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    subset: str | list[str],
    corruptions: list[str],
    include_clean: bool = True,
    seed: int = 0,
    max_examples: int | None = None,
    normalization_dataset: str = "svhn",
) -> DataLoader:
    return make_digitrobust_loader(
        MixedDigitRobustDataset(
            root=data_dir,
            subset=subset,
            corruptions=corruptions,
            include_clean=include_clean,
            seed=seed,
            max_examples=max_examples,
            normalization_dataset=normalization_dataset,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
    )


def ensure_digitrobust_files(
    data_dir: str | Path,
    subset: str | list[str],
    corruptions: list[str],
) -> None:
    root = Path(data_dir)
    subsets = [subset] if isinstance(subset, str) else list(subset)
    required = []
    for subset_name in subsets:
        required.append(root / "clean" / subset_name / "test" / "images.npy")
        required.extend(
            root / "corrupt" / f"{subset_name}-corrupt" / corruption / "images.npy"
            for corruption in corruptions
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing DigitRobust files:\n"
            f"{missing_list}\n\n"
            "Download the dataset first with:\n"
            "  bash tools/download_digitrobust.sh"
        )
