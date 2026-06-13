# Robustness-Preserving Test-Time Adaptation

This repository contains the code for the paper project on preserving corruption robustness under domain shift.

## Documentation

- Project structure: `CODE_STRUCTURE.md`
- Paper table reproduction commands: `RUN_EXPERIMENTS.md`

## Our Method

**RPT and SARPT are the methods proposed and developed by our group for this
project.** Their implementation is kept in `rpt_sarpt/`.

AugMix, AdaBN, TENT, EATA, and CoTTA are comparison methods. Their
implementations are grouped under `baseline/` to distinguish established baselines from the
project's proposed methods.

## Setup

```bash
pip install -r requirements.txt
```

## Download Results and Datasets

Use Git Bash, WSL, or Linux shell for the download scripts.

Download pretrained checkpoints and experiment results:

```bash
bash tools/download_outputs.sh
```

This extracts files under:

```text
outputs/
```

Download the required custom datasets:

```bash
bash tools/download_digitrobust.sh
bash tools/download_cifar10_1_c.sh
```

These extract files under:

```text
datasets/DigitRobust/
datasets/CIFAR-10.1-C-small/
```

CIFAR-10-C, MNIST-C, CIFAR-10.1, and torchvision datasets are handled by their
data loaders when needed.

## Main Commands

Train or evaluate experiments from YAML configs:

```bash
python -m entry_point.run_experiment <config.yaml>
```

Train a simple baseline directly:

```bash
python -m entry_point.train
```

Evaluate one checkpoint directly:

```bash
python -m entry_point.evaluate --checkpoint <path-to-best.pt>
```
