# Paper Table Reproduction Commands

To reproduce every paper table with five seeds, run the protocol-aware runner
from the repository root:

```bash
bash tools/run_paper_tables.sh
```

This script also runs `python -m analysis.analyze_results_two` automatically at
the end. If required datasets are missing, it downloads them before the
preflight checkpoint check.

If checkpoints are missing, download the prepared outputs first. The downloaded
outputs include the pretrained checkpoints used by the table configs:

```bash
bash tools/download_outputs.sh
```

Alternatively, train the source models yourself:

```bash
python -m entry_point.run_experiment \
  configs/experiment/training/baseline.yaml \
  configs/experiment/training/augmix.yaml \
  configs/experiment/training/svhn_baseline.yaml \
  configs/experiment/training/svhn_augmix.yaml \
  configs/experiment/training/mnist_baseline.yaml \
  configs/experiment/training/mnist_augmix.yaml
```

To rebuild the table outputs from existing results without rerunning
experiments, run:

```bash
python -m analysis.analyze_results_two
```

## How To Tune YAML

Usually edit these files:

- `configs/dataset/*.yaml`: dataset size, corruptions, seeds, mixed stream length.
- `configs/test_adapt/*.yaml`: TTA/RPT/SARPT hyperparameters.
- `configs/experiment/table/<setting>/<method>.yaml`: checkpoint and output folder.
- `configs/experiment/training/*.yaml`: source-model training settings.

## Table 1: `tab:single_eval_size`

This table reports dataset sizes from the evaluation protocol. It does not require a training or evaluation command.

Relevant YAML files:

```text
configs/dataset/digitrobust_mnistc.yaml
configs/dataset/digitrobust_svhnc.yaml
```

## Table 2: `tab:standard_tta`

Paper table: digit-domain Single and Mixed transfer results.

```bash
python -m entry_point.run_experiment --protocol episodic \
  configs/experiment/table/svhn_to_svhn_single/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_single/adabn.yaml \
  configs/experiment/table/svhn_to_svhn_single/tent.yaml \
  configs/experiment/table/svhn_to_svhn_single/eata.yaml \
  configs/experiment/table/svhn_to_svhn_single/rpt.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/adabn.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/tent.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/eata.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/rpt.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_single/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_single/adabn.yaml \
  configs/experiment/table/mnist_to_mnist_single/tent.yaml \
  configs/experiment/table/mnist_to_mnist_single/eata.yaml \
  configs/experiment/table/mnist_to_mnist_single/rpt.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/adabn.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/tent.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/eata.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/rpt.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_single/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_single/adabn.yaml \
  configs/experiment/table/svhn_to_mnist_single/tent.yaml \
  configs/experiment/table/svhn_to_mnist_single/eata.yaml \
  configs/experiment/table/svhn_to_mnist_single/rpt.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/adabn.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/eata.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/rpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_single/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_single/adabn.yaml \
  configs/experiment/table/mnist_to_svhn_single/tent.yaml \
  configs/experiment/table/mnist_to_svhn_single/eata.yaml \
  configs/experiment/table/mnist_to_svhn_single/rpt.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/adabn.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/tent.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/eata.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/rpt.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_rpt.yaml
```

## Table 3: `tab:continual_short`

Paper table: 30,000-image continual mixed-corruption streams.

```bash
python -m entry_point.run_experiment --protocol continual_short \
  configs/experiment/table/svhn_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_cotta.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_sarpt.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_cotta.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_sarpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_cotta.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_sarpt.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_cotta.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_sarpt.yaml
```

## Table 4: `tab:continual_long`

Paper table: 100,000-image long continual streams.

```bash
python -m entry_point.run_experiment --protocol continual_long \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/augmix_cotta.yaml \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_svhn_mnist_mixed/augmix_sarpt.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/augmix_cotta.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_mnist_mixed/augmix_sarpt.yaml
```

Build the long-stream figures:

```bash
python -m analysis.plot_long_stream_accuracy
```

## Table 5: `tab:ablation_jsd`

Paper table: JSD consistency ablation on SVHN -> MNIST mixed corruptions.

```bash
python -m entry_point.run_experiment --protocol episodic \
  configs/experiment/table/svhn_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_rpt.yaml
```

## Table 6: `tab:ablation_anchor`

Paper table: source-anchor ablation on SVHN -> MNIST continual mixed corruptions.

```bash
python -m entry_point.run_experiment --protocol continual_short \
  configs/experiment/table/svhn_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_sarpt.yaml
```

## Table 7: `tab:cifar`

Paper table: CIFAR-10-C and synthetic CIFAR-10.1-C-small results.

```bash
python -m entry_point.run_experiment --protocol episodic \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/baseline.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/augmix.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/augmix_adabn.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/augmix_eata.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/augmix_tent.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_single/augmix_rpt.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/baseline.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/augmix.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/augmix_adabn.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/augmix_eata.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/augmix_tent.yaml \
  configs/experiment/table/cifar10_to_cifar10_1_c_mixed/augmix_rpt.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/baseline.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/augmix.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/augmix_adabn.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/augmix_eata.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/augmix_tent.yaml \
  configs/experiment/table/cifar10_to_cifar10_single/augmix_rpt.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/baseline.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/augmix.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/augmix_adabn.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/augmix_eata.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/augmix_tent.yaml \
  configs/experiment/table/cifar10_to_cifar10_mixed/augmix_rpt.yaml
```

## Table 8: `tab:crr_digit`

Paper table: CRR under digit-domain shifts.

Run the clean, single, and mixed digit results needed by CRR:

```bash
python -m entry_point.run_experiment --protocol episodic \
  configs/experiment/table/svhn_to_svhn_clean/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_clean/augmix.yaml \
  configs/experiment/table/svhn_to_svhn_clean/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_svhn_clean/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_clean/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_clean/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_svhn_single/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_single/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_svhn_mixed/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_clean/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_clean/augmix.yaml \
  configs/experiment/table/svhn_to_mnist_clean/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_mnist_clean/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_clean/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_clean/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_single/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_single/augmix_rpt.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_adabn.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/svhn_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_clean/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_clean/augmix.yaml \
  configs/experiment/table/mnist_to_mnist_clean/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_mnist_clean/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_clean/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_clean/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_single/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_single/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_mnist_mixed/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_clean/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_clean/augmix.yaml \
  configs/experiment/table/mnist_to_svhn_clean/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_svhn_clean/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_clean/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_clean/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_single/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_single/augmix_rpt.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/baseline.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_adabn.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_tent.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_eata.yaml \
  configs/experiment/table/mnist_to_svhn_mixed/augmix_rpt.yaml
```

`tools/run_paper_tables.sh` builds the accuracy table automatically. If you are
using existing results, rebuild the accuracy table first, then compute CRR:

```bash
python -m analysis.analyze_results_two

python -m analysis.analyze_crr
```
