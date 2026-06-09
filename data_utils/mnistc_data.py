from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_utils.data import build_test_transform, build_torchvision_dataset


MNISTC_DOWNLOAD_URL = "https://zenodo.org/records/3239543/files/mnist_c.zip?download=1"
MNISTC_ARCHIVE_NAME = "mnist_c.zip"
MNISTC_CORRUPTIONS = (
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
)


def mnistc_required_files(data_dir: str | Path, corruptions: list[str]) -> list[Path]:
    root = Path(data_dir)
    required: list[Path] = []
    for corruption in corruptions:
        corruption_root = root / corruption
        required.append(corruption_root / "test_images.npy")
        required.append(corruption_root / "test_labels.npy")
    return required


def missing_mnistc_files(data_dir: str | Path, corruptions: list[str]) -> list[Path]:
    return [path for path in mnistc_required_files(data_dir, corruptions) if not path.exists()]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading MNIST-C from {url}")
    print(f"Saving archive to {destination}")
    try:
        with urllib.request.urlopen(url) as response, temp_destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as exc:
        temp_destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download MNIST-C from {url}: {exc}") from exc

    temp_destination.replace(destination)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()
            try:
                target_path.relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to extract unsafe MNIST-C archive member: {member.filename}"
                ) from exc
        archive.extractall(destination)


def ensure_mnistc_files(data_dir: str | Path, corruptions: list[str]) -> None:
    missing = missing_mnistc_files(data_dir, corruptions)
    if not missing:
        return

    data_root = Path(data_dir).parent
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / MNISTC_ARCHIVE_NAME
    print("Missing MNIST-C files:")
    for path in missing:
        print(f"  - {path}")

    if archive_path.exists() and zipfile.is_zipfile(archive_path):
        print(f"Using existing MNIST-C archive: {archive_path}")
    else:
        if archive_path.exists():
            print(f"Replacing incomplete MNIST-C archive: {archive_path}")
        _download_file(MNISTC_DOWNLOAD_URL, archive_path)

    print(f"Extracting MNIST-C archive into {data_root}")
    _safe_extract_zip(archive_path, data_root)

    missing_after_extract = missing_mnistc_files(data_dir, corruptions)
    if missing_after_extract:
        missing_list = "\n".join(f"  - {path}" for path in missing_after_extract)
        raise FileNotFoundError(
            "MNIST-C download/extraction completed, but required files are still missing:\n"
            f"{missing_list}\n\n"
            f"Expected files under: {Path(data_dir)}"
        )


class MNISTCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int = 1,
        normalization_dataset: str = "svhn",
    ) -> None:
        if severity != 1:
            raise ValueError("MNIST-C does not provide severity blocks; use severity 1.")

        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = build_test_transform(
            dataset_name="mnistc",
            normalization_dataset=normalization_dataset,
        )

        corruption_root = self.root / corruption
        image_path = corruption_root / "test_images.npy"
        labels_path = corruption_root / "test_labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing MNIST-C images file: {image_path}.")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing MNIST-C labels file: {labels_path}.")

        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"MNIST-C images/labels length mismatch for {corruption}: "
                f"{len(self.images)} != {len(self.labels)}."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = np.asarray(self.images[index], dtype=np.uint8)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
        image_pil = Image.fromarray(image, mode="L")
        label = int(self.labels[index])
        return self.transform(image_pil), label


class MixedMNISTCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        corruptions: list[str],
        clean_data_dir: str = "datasets",
        clean_download: bool = False,
        include_clean: bool = True,
        seed: int = 0,
        max_examples: int | None = None,
        normalization_dataset: str = "svhn",
    ) -> None:
        if not corruptions and not include_clean:
            raise ValueError("Mixed MNIST-C requires at least one corruption.")

        self.datasets: list[Dataset] = []
        if include_clean:
            self.datasets.append(
                build_torchvision_dataset(
                    dataset_name="mnist",
                    data_dir=clean_data_dir,
                    train=False,
                    transform=build_test_transform(
                        dataset_name="mnist",
                        normalization_dataset=normalization_dataset,
                    ),
                    download=clean_download,
                )
            )

        self.datasets.extend(
            MNISTCDataset(
                root=root,
                corruption=corruption,
                severity=1,
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


def build_mnistc_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruption: str,
    severity: int = 1,
    normalization_dataset: str = "svhn",
) -> DataLoader:
    dataset = MNISTCDataset(
        root=data_dir,
        corruption=corruption,
        severity=severity,
        normalization_dataset=normalization_dataset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_mixed_mnistc_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    corruptions: list[str],
    clean_data_dir: str = "datasets",
    clean_download: bool = False,
    include_clean: bool = True,
    seed: int = 0,
    max_examples: int | None = None,
    normalization_dataset: str = "svhn",
) -> DataLoader:
    dataset = MixedMNISTCDataset(
        root=data_dir,
        corruptions=corruptions,
        clean_data_dir=clean_data_dir,
        clean_download=clean_download,
        include_clean=include_clean,
        seed=seed,
        max_examples=max_examples,
        normalization_dataset=normalization_dataset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
