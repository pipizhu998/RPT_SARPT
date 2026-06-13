# Code Structure
This file contains AI-written comments to help explain and understand the structure of the codebase.

## Root Files

```text
.gitattributes         Git text/binary normalization rules.
.gitignore            Ignored local caches, archives, datasets, and outputs.
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
```

## Config Files

```text
configs/model/                         Model architecture fragments.
configs/train_aug/                     Source-training augmentation fragments.
configs/test_adapt/                    Test-time adaptation method fragments.
configs/dataset/                       Dataset, corruption, clean-data, and stream fragments.
configs/protocol/                      Episodic and continual evaluation protocols.
configs/experiment/training/           Source-model training experiment configs.
configs/experiment/table/<setting>/    Paper-table experiment configs; folder names encode the source-target setting, and file names encode the method.
```

## Testing

```text
testing/.gitignore                    Ignores temporary test artifacts.
testing/README.md                     Unit-test documentation.
testing/__init__.py                   Package marker.
testing/_bootstrap.py                 Test runtime bootstrap helpers.
testing/run_tests.py                  Unittest runner.
testing/test_analysis_utilities.py    Tests analysis parsing and aggregation helpers.
testing/test_core_utils.py            Tests shared runtime utilities.
testing/test_evaluation_utilities.py  Tests evaluation path, cache-key, and recorder helpers.
testing/test_experiment_config.py     Tests config loading and merging.
```

## Data And Outputs

```text
datasets/       Downloaded datasets used by loaders; only `.gitkeep` is tracked.
outputs/        Checkpoints, raw experiment results, tables, and figures; only `.gitkeep` is tracked.
.tmp/           Temporary download working directory ignored by Git.
```
