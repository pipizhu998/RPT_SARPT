from __future__ import annotations

import tarfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_utils.data import CIFAR10_MEAN, CIFAR10_STD, build_torchvision_dataset


CIFAR10C_DOWNLOAD_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_ARCHIVE_NAME = "CIFAR-10-C.tar"
CIFAR10C_CORRUPTIONS = (
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


def cifar10c_required_files(data_dir: str | Path, corruptions: list[str]) -> list[Path]:
    root = Path(data_dir)
    required = [root / "labels.npy"]
    required.extend(root / f"{corruption}.npy" for corruption in corruptions)
    return required


def missing_cifar10c_files(data_dir: str | Path, corruptions: list[str]) -> list[Path]:
    return [path for path in cifar10c_required_files(data_dir, corruptions) if not path.exists()]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading CIFAR-10-C from {url}")
    print(f"Saving archive to {destination}")
    try:
        with urllib.request.urlopen(url) as response, temp_destination.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total_bytes = int(total_header) if total_header else None
            downloaded = 0
            next_report = 64 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total_bytes:
                        print(
                            f"Downloaded {downloaded / 1024 / 1024:.0f} MiB "
                            f"of {total_bytes / 1024 / 1024:.0f} MiB"
                        )
                    else:
                        print(f"Downloaded {downloaded / 1024 / 1024:.0f} MiB")
                    next_report += 64 * 1024 * 1024
    except urllib.error.URLError as exc:
        temp_destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download CIFAR-10-C from {url}: {exc}") from exc

    temp_destination.replace(destination)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            target_path = (destination / member.name).resolve()
            try:
                target_path.relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to extract unsafe CIFAR-10-C archive member: {member.name}"
                ) from exc
        archive.extractall(destination)


def ensure_cifar10c_files(data_dir: str | Path, corruptions: list[str]) -> None:
    missing = missing_cifar10c_files(data_dir, corruptions)
    if not missing:
        return

    data_root = Path(data_dir).parent
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / CIFAR10C_ARCHIVE_NAME
    print("Missing CIFAR-10-C files:")
    for path in missing:
        print(f"  - {path}")

    downloaded_archive = False
    if archive_path.exists() and tarfile.is_tarfile(archive_path):
        print(f"Using existing CIFAR-10-C archive: {archive_path}")
    else:
        if archive_path.exists():
            print(f"Replacing incomplete CIFAR-10-C archive: {archive_path}")
        _download_file(CIFAR10C_DOWNLOAD_URL, archive_path)
        downloaded_archive = True

    print(f"Extracting CIFAR-10-C archive into {data_root}")
    _safe_extract_tar(archive_path, data_root)

    missing_after_extract = missing_cifar10c_files(data_dir, corruptions)
    if missing_after_extract:
        missing_list = "\n".join(f"  - {path}" for path in missing_after_extract)
        raise FileNotFoundError(
            "CIFAR-10-C download/extraction completed, but required files are still missing:\n"
            f"{missing_list}\n\n"
            f"Expected files under: {Path(data_dir)}"
        )
    if downloaded_archive:
        try:
            archive_path.unlink()
        except OSError as exc:
            print(f"Could not remove downloaded CIFAR-10-C archive {archive_path}: {exc}")
        else:
            print(f"Removed downloaded CIFAR-10-C archive: {archive_path}")


def cifar10c_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def sample_condition_indices(
    total_examples: int,
    max_examples: int | None,
    seed: int,
) -> np.ndarray:
    indices = np.arange(total_examples, dtype=np.int64)
    if max_examples is None:
        return indices
    if max_examples <= 0:
        raise ValueError("max_examples_per_condition must be positive when provided.")
    if max_examples >= total_examples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_examples, replace=False))


class CIFAR10CDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        max_examples: int | None = None,
        seed: int = 0,
    ) -> None:
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("CIFAR-10-C severity must be one of 1, 2, 3, 4, 5.")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = cifar10c_transform()

        image_path = self.root / f"{corruption}.npy"
        labels_path = self.root / "labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10-C file: {image_path}. "
                "Put the official CIFAR-10-C .npy files under datasets/CIFAR-10-C/."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10-C labels file: {labels_path}."
            )

        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.start = (severity - 1) * 10000
        self.end = severity * 10000
        self.indices = self.start + sample_condition_indices(
            total_examples=10000,
            max_examples=max_examples,
            seed=seed,
        )

        if len(self.images) < self.end:
            raise ValueError(
                f"{image_path} has {len(self.images)} images, "
                f"but severity {severity} needs at least {self.end}."
            )
        if len(self.labels) not in {10000, len(self.images)}:
            raise ValueError(
                "labels.npy must contain either 10000 labels or one label per image."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_index = int(self.indices[index])
        image = Image.fromarray(np.asarray(self.images[image_index], dtype=np.uint8))
        label_index = image_index if len(self.labels) == len(self.images) else image_index - self.start
        label = int(self.labels[label_index])
        return self.transform(image), label


class MixedCIFAR10CDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruptions: list[str],
        severity_levels: list[int],
        clean_data_dir: str = "datasets",
        clean_download: bool = False,
        include_clean: bool = False,
        seed: int = 0,
        max_examples: int | None = None,
        max_examples_per_condition: int | None = None,
    ) -> None:
        if not corruptions and not include_clean:
            raise ValueError("Mixed CIFAR-10-C requires at least one corruption.")
        if corruptions and not severity_levels:
            raise ValueError("Mixed CIFAR-10-C requires at least one severity level.")
        invalid_severities = sorted(set(severity_levels) - {1, 2, 3, 4, 5})
        if invalid_severities:
            raise ValueError(
                "CIFAR-10-C severity must be one of 1, 2, 3, 4, 5; "
                f"got {invalid_severities}."
            )

        self.root = Path(root)
        self.corruptions = list(corruptions)
        self.severity_levels = list(severity_levels)
        self.transform = cifar10c_transform()
        self.include_clean = include_clean

        labels_path = self.root / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C labels file: {labels_path}.")

        self.images_by_corruption = {}
        for corruption in self.corruptions:
            image_path = self.root / f"{corruption}.npy"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing CIFAR-10-C file: {image_path}. "
                    "Put the official CIFAR-10-C .npy files under datasets/CIFAR-10-C/."
                )
            images = np.load(image_path, mmap_mode="r")
            if len(images) < 50000:
                raise ValueError(
                    f"{image_path} has {len(images)} images, but CIFAR-10-C "
                    "mixed evaluation needs all five 10000-image severity blocks."
                )
            self.images_by_corruption[corruption] = images

        self.labels = np.load(labels_path, mmap_mode="r")
        if len(self.labels) not in {10000, 50000}:
            raise ValueError("labels.npy must contain either 10000 or 50000 labels.")

        local_indices = sample_condition_indices(
            total_examples=10000,
            max_examples=max_examples_per_condition,
            seed=seed,
        )
        samples: list[tuple[int, int, int]] = []
        if include_clean:
            samples.extend((-1, int(local_index), int(local_index)) for local_index in local_indices)
        for corruption_index, _corruption in enumerate(self.corruptions):
            for severity in self.severity_levels:
                offset = (severity - 1) * 10000
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
                raise ValueError("max_examples must be positive when provided.")
            order = order[:max_examples]
        self.order = order
        self.clean_dataset = None
        if include_clean:
            self.clean_dataset = build_torchvision_dataset(
                dataset_name="cifar10",
                data_dir=clean_data_dir,
                train=False,
                transform=self.transform,
                download=clean_download,
            )

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample_index = int(self.order[index])
        corruption_index, cifar10c_index, local_index = self.samples[sample_index]
        if corruption_index < 0:
            if self.clean_dataset is None:
                raise RuntimeError("Clean CIFAR-10 dataset was not initialized.")
            return self.clean_dataset[local_index]

        corruption = self.corruptions[corruption_index]

        image = Image.fromarray(
            np.asarray(self.images_by_corruption[corruption][cifar10c_index], dtype=np.uint8)
        )
        label_index = cifar10c_index if len(self.labels) == 50000 else local_index
        label = int(self.labels[label_index])
        return self.transform(image), label


def build_cifar10c_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruption: str,
    severity: int,
    max_examples: int | None = None,
    seed: int = 0,
) -> DataLoader:
    dataset = CIFAR10CDataset(
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


def build_mixed_cifar10c_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruptions: list[str],
    severity_levels: list[int],
    clean_data_dir: str = "datasets",
    clean_download: bool = False,
    include_clean: bool = False,
    seed: int = 0,
    max_examples: int | None = None,
    max_examples_per_condition: int | None = None,
) -> DataLoader:
    dataset = MixedCIFAR10CDataset(
        root=data_dir,
        corruptions=corruptions,
        severity_levels=severity_levels,
        clean_data_dir=clean_data_dir,
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
