from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data_utils.data import build_test_transform, make_loader


CIFAR10_1_BASE_URL = (
    "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets"
)
CIFAR10_1_VERSIONS = ("v4", "v6")
CIFAR10_1_FILE_NAMES = {
    "v4": ("cifar10.1_v4_data.npy", "cifar10.1_v4_labels.npy"),
    "v6": ("cifar10.1_v6_data.npy", "cifar10.1_v6_labels.npy"),
}


def cifar10_1_file_paths(
    data_dir: str | Path,
    version: str = "v6",
) -> tuple[Path, Path]:
    if version not in CIFAR10_1_FILE_NAMES:
        raise ValueError(
            f"Unsupported CIFAR-10.1 version: {version}. "
            f"Expected one of {CIFAR10_1_VERSIONS}."
        )
    data_file, labels_file = CIFAR10_1_FILE_NAMES[version]
    root = Path(data_dir)
    return root / data_file, root / labels_file


def missing_cifar10_1_files(
    data_dir: str | Path,
    version: str = "v6",
) -> list[Path]:
    return [path for path in cifar10_1_file_paths(data_dir, version) if not path.exists()]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading CIFAR-10.1 file from {url}")
    print(f"Saving to {destination}")
    try:
        with urllib.request.urlopen(url) as response, temp_destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as exc:
        temp_destination.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download CIFAR-10.1 file from {url}: {exc}") from exc

    temp_destination.replace(destination)


def ensure_cifar10_1_files(
    data_dir: str | Path,
    version: str = "v6",
    download: bool = True,
) -> None:
    missing = missing_cifar10_1_files(data_dir, version)
    if not missing:
        return

    if not download:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing CIFAR-10.1 files:\n"
            f"{missing_list}\n\n"
            "Either put the official .npy files there, or set clean_download: true."
        )

    data_file, labels_file = CIFAR10_1_FILE_NAMES[version]
    for file_name in (data_file, labels_file):
        path = Path(data_dir) / file_name
        if path.exists():
            continue
        _download_file(f"{CIFAR10_1_BASE_URL}/{file_name}", path)

    missing_after_download = missing_cifar10_1_files(data_dir, version)
    if missing_after_download:
        missing_list = "\n".join(f"  - {path}" for path in missing_after_download)
        raise FileNotFoundError(
            "CIFAR-10.1 download completed, but required files are still missing:\n"
            f"{missing_list}"
        )


class CIFAR101Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        version: str = "v6",
        download: bool = True,
        normalization_dataset: str = "cifar10",
    ) -> None:
        ensure_cifar10_1_files(root, version=version, download=download)
        data_path, labels_path = cifar10_1_file_paths(root, version)

        self.images = np.load(data_path)
        self.labels = np.load(labels_path)
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"CIFAR-10.1 image/label length mismatch: "
                f"{len(self.images)} images vs {len(self.labels)} labels."
            )
        if self.images.ndim != 4 or self.images.shape[1:] != (32, 32, 3):
            raise ValueError(
                f"Expected CIFAR-10.1 images with shape (N, 32, 32, 3), "
                f"got {self.images.shape}."
            )

        self.transform = build_test_transform(
            dataset_name="cifar10",
            normalization_dataset=normalization_dataset,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(np.asarray(self.images[index], dtype=np.uint8))
        label = int(self.labels[index])
        return self.transform(image), label


def build_cifar10_1_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    version: str = "v6",
    download: bool = True,
    normalization_dataset: str = "cifar10",
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    dataset = CIFAR101Dataset(
        root=data_dir,
        version=version,
        download=download,
        normalization_dataset=normalization_dataset,
    )
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(dataset)).tolist()
        dataset = Subset(dataset, order)
    return make_loader(dataset, batch_size, num_workers, shuffle=False)
