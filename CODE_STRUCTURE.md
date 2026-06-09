# Code Structure
This file contains AI-written comments to help explain and understand the structure of the codebase.## Root Files

```text
README.md             Main setup, download, and run instructions.
RUN_EXPERIMENTS.md    Detailed notes for running experiments and changing YAML.
CODE_STRUCTURE.md     Short map of the repository.
requirements.txt      Python package dependencies.
```

## Pipeline Entrypoints

```text
entry_point/__init__.py          Package marker for runnable entry points.
entry_point/train.py             Direct training entry point for source models.
entry_point/evaluate.py          Evaluation entry point for clean, corrupted, mixed, and continual streams.
entry_point/run_experiment.py    YAML experiment runner for training and evaluation jobs.
```

## Core Code

```text
core/__init__.py                    Package marker.
core/experiment_config.py           Loads YAML fragments and merges composed experiment configs.
core/models.py                      CNN and ResNet model definitions used by training and evaluation.
core/test_time_adapt.py             Dispatches test-time adaptation methods.
core/training_method_interface.py   Builds training loaders, models, optimizers, and output paths from YAML.
core/training_method_train.py       Runs one training job from a TrainingMethodInterface.
core/utils.py                       Shared runtime helpers, metrics, CSV/JSON saving, and clean evaluation.
```

## Data Loading

```text
data_utils/__init__.py              Package marker.
data_utils/data.py                  Clean CIFAR-10, MNIST, and SVHN loaders and transforms.
data_utils/AugMix_data.py           Compatibility wrapper for AugMix training data helpers.
data_utils/cifar10_1_data.py        CIFAR-10.1 download and loader.
data_utils/cifar10_1_c_data.py      CIFAR-10.1-C-small loader for the downloaded dataset.
data_utils/cifar10c_data.py         CIFAR-10-C download and loaders.
data_utils/digitrobust_data.py      DigitRobust clean, corrupted, and mixed loaders.
data_utils/mnistc_data.py           MNIST-C download and loaders.
```

## Proposed Method

```text
rpt_sarpt/__init__.py       Public exports for RPT and SARPT.
rpt_sarpt/methods.py        RPT/SARPT model wrapper, losses, optimizer setup, and BatchNorm adaptation.
rpt_sarpt/evaluation.py     RPT/SARPT evaluation wrapper and optional AugMix precompute cache.
```

## Baselines

```text
baseline/           Borrowed comparison implementations.
baseline/adabn/     AdaBN baseline.
baseline/augmix/    AugMix baseline.
baseline/cotta/     CoTTA baseline.
baseline/eata/      EATA baseline.
baseline/tent/      TENT baseline.
```

Files inside `baseline/` are not expanded here because they are reference code borrowed for comparison.

## Analysis Scripts

```text
analysis/__init__.py                         Package marker.
analysis/analyze_results.py                  Older result summary tables and plots.
analysis/analyze_results_two.py              Main paper table builder with mean and std aggregation.
analysis/analyze_results_sweep.py            Sweep summary, best-config CSV, and heatmap output.
analysis/analyze_crr.py                      CRR table and plot generation.
analysis/plot_collapse_probability.py        Probability-log plots for prediction collapse checks.
analysis/plot_long_stream_accuracy.py        Continual-stream accuracy curves.
analysis/plot_per_corruption_accuracy.py     Per-corruption accuracy bar plots.
```

## Tools

```text
tools/download_outputs.sh        Downloads outputs, checkpoints, and saved results.
tools/download_digitrobust.sh    Downloads DigitRobust into datasets/DigitRobust.
tools/download_cifar10_1_c.sh    Downloads CIFAR-10.1-C-small into datasets/CIFAR-10.1-C-small.
tools/run_paper_tables.sh        Runs the paper table configs and rebuilds analysis outputs.
tools/update_outputs.sh          Uploads the current outputs archive through rclone.
tools/update_outputs.conf        Settings for update_outputs.sh.
tools/update_digitrobust.sh      Uploads the DigitRobust archive through rclone.
tools/update_digitrobust.conf    Settings for update_digitrobust.sh.
tools/update_cifar10_1_c.sh      Uploads the CIFAR-10.1-C-small archive through rclone.
tools/update_cifar10_1_c.conf    Settings for update_cifar10_1_c.sh.
```

## Config Files

```text
configs/model/resnet18.yaml                  ResNet-18 model setting.

configs/train_aug/none.yaml                  Clean training.
configs/train_aug/augmix.yaml                AugMix training.

configs/test_adapt/none.yaml                 No test-time adaptation.
configs/test_adapt/adabn.yaml                AdaBN setting.
configs/test_adapt/tent.yaml                 TENT setting.
configs/test_adapt/eata.yaml                 EATA setting.
configs/test_adapt/cotta.yaml                CoTTA setting.
configs/test_adapt/rpt.yaml                  RPT setting.
configs/test_adapt/sarpt.yaml                SARPT setting.

configs/protocol/episodic.yaml               Episodic paper protocol.
configs/protocol/continual_short.yaml        Short continual paper protocol.
configs/protocol/continual_long.yaml         Long continual paper protocol.
configs/protocol/continual_long_prob_log.yaml Long continual protocol with probability logging.
```

```text
configs/dataset/cifar10.yaml                                      Clean CIFAR-10.
configs/dataset/cifar10_1.yaml                                    CIFAR-10.1.
configs/dataset/cifar10_1_c.yaml                                  CIFAR-10.1-C-small single-condition stream.
configs/dataset/cifar10_1_c_mixed.yaml                            CIFAR-10.1-C-small mixed stream.
configs/dataset/cifar10c.yaml                                     CIFAR-10-C single-condition stream.
configs/dataset/cifar10c_mixed.yaml                               CIFAR-10-C mixed stream.
configs/dataset/mnist.yaml                                        Clean MNIST.
configs/dataset/mnistc.yaml                                       MNIST-C single-condition stream.
configs/dataset/mnistc_mixed.yaml                                 MNIST-C mixed stream.
configs/dataset/svhn.yaml                                         Clean SVHN.
configs/dataset/digitrobust_mnistc.yaml                           DigitRobust MNIST-C stream.
configs/dataset/digitrobust_mnistc_mixed.yaml                     Mixed DigitRobust MNIST-C stream.
configs/dataset/digitrobust_mnistc_source_mnist.yaml              DigitRobust MNIST-C with MNIST source normalization.
configs/dataset/digitrobust_mnistc_source_mnist_mixed.yaml        Mixed version of the MNIST-source setting.
configs/dataset/digitrobust_svhnc.yaml                            DigitRobust SVHN-C stream.
configs/dataset/digitrobust_svhnc_mixed.yaml                      Mixed DigitRobust SVHN-C stream.
configs/dataset/digitrobust_svhnc_source_mnist.yaml               DigitRobust SVHN-C with MNIST source normalization.
configs/dataset/digitrobust_svhnc_source_mnist_mixed.yaml         Mixed version of the MNIST-source SVHN-C setting.
configs/dataset/digitrobust_svhnc_mnistc_mixed.yaml               Mixed SVHN-C and MNIST-C stream.
configs/dataset/digitrobust_svhnc_mnistc_source_mnist_mixed.yaml  Mixed SVHN-C and MNIST-C stream with MNIST source normalization.
```

```text
configs/experiment/training/baseline.yaml          CIFAR-10 clean source training.
configs/experiment/training/baseline_lr0p01.yaml   CIFAR-10 clean training with lr 0.01.
configs/experiment/training/baseline_smoke.yaml    Small clean training smoke test.
configs/experiment/training/augmix.yaml            CIFAR-10 AugMix source training.
configs/experiment/training/augmix_lr0p01.yaml     CIFAR-10 AugMix training with lr 0.01.
configs/experiment/training/augmix_smoke.yaml      Small AugMix training smoke test.
configs/experiment/training/mnist_baseline.yaml    MNIST clean source training.
configs/experiment/training/mnist_augmix.yaml      MNIST AugMix source training.
configs/experiment/training/svhn_baseline.yaml     SVHN clean source training.
configs/experiment/training/svhn_augmix.yaml       SVHN AugMix source training.
```

`configs/experiment/table/<setting>/<method>.yaml` contains concrete table runs. The folder name gives the source-target setting, and the file name gives the training/adaptation method.

```text
cifar10_to_cifar10_1_c_mixed/      adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
cifar10_to_cifar10_1_c_single/     adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
cifar10_to_cifar10_1_clean/        adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
cifar10_to_cifar10_clean/          adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
cifar10_to_cifar10_mixed/          adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
cifar10_to_cifar10_single/         adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_mnist_clean/              adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_mnist_mixed/              adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_mnist_single/             adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_svhn_clean/               adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_svhn_mixed/               adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_svhn_mnist_mixed/         adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
mnist_to_svhn_single/              adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_mnist_clean/               adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_mnist_mixed/               adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_mnist_single/              adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_svhn_clean/                adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_svhn_mixed/                adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_svhn_mnist_mixed/          adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
svhn_to_svhn_single/               adabn, augmix, augmix_adabn, augmix_cotta, augmix_eata, augmix_rpt, augmix_sarpt, augmix_tent, baseline, cotta, eata, rpt, sarpt, tent.
```

## Data And Outputs

```text
datasets/       Downloaded datasets used by loaders.
outputs/        Checkpoints, raw experiment results, tables, and figures.
.tmp/           Temporary download working directory.
```
