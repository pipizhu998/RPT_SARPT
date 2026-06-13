#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

seeds=(0 1 2 3 4)
devices=()
dry_run=()
python_cmd=()

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=(${PYTHON_BIN})
elif command -v python.exe >/dev/null 2>&1; then
  python_cmd=(python.exe)
elif command -v python >/dev/null 2>&1; then
  python_cmd=(python)
elif command -v python3 >/dev/null 2>&1; then
  python_cmd=(python3)
elif command -v py.exe >/dev/null 2>&1; then
  python_cmd=(py.exe -3)
elif command -v py >/dev/null 2>&1; then
  python_cmd=(py -3)
else
  echo "Missing Python command. Set PYTHON_BIN or install python/python3." >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --devices)
      devices=(--devices "$2")
      shift 2
      ;;
    --dry-run)
      dry_run=(--dry-run)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: tools/run_paper_tables.sh [--devices cuda:0,cuda:1] [--dry-run]" >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_cifar10_1() {
  local data_dir="datasets/CIFAR-10.1"
  local base_url="https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets"
  local file
  require_command curl
  mkdir -p "${data_dir}"
  for file in cifar10.1_v6_data.npy cifar10.1_v6_labels.npy; do
    if [[ ! -s "${data_dir}/${file}" ]]; then
      echo "Downloading CIFAR-10.1 ${file}"
      rm -f "${data_dir}/${file}.part"
      curl -L --fail --retry 3 --connect-timeout 20 \
        -o "${data_dir}/${file}.part" \
        "${base_url}/${file}"
      mv "${data_dir}/${file}.part" "${data_dir}/${file}"
    fi
  done
}

ensure_paper_datasets() {
  if [[ ! -d datasets/DigitRobust ]]; then
    echo "Missing datasets/DigitRobust; downloading it now."
    bash tools/download_digitrobust.sh
  fi

  if [[ ! -s datasets/CIFAR-10.1/cifar10.1_v6_data.npy \
    || ! -s datasets/CIFAR-10.1/cifar10.1_v6_labels.npy ]]; then
    echo "Missing CIFAR-10.1 v6 files; downloading them now."
    download_cifar10_1
  fi

  if [[ ! -d datasets/CIFAR-10.1-C-small ]]; then
    echo "Missing datasets/CIFAR-10.1-C-small; downloading it now."
    bash tools/download_cifar10_1_c.sh
  fi
}

preflight() {
  local missing=()
  local path
  for path in \
    outputs/experiments/training/clean/baseline_resnet18/best.pt \
    outputs/experiments/training/augmix/augmix_resnet18/best.pt \
    outputs/experiments/training/clean/svhn_baseline_resnet18/best.pt \
    outputs/experiments/training/augmix/svhn_augmix_resnet18/best.pt \
    outputs/experiments/training/clean/mnist_baseline_resnet18/best.pt \
    outputs/experiments/training/augmix/mnist_augmix_resnet18/best.pt
  do
    if [[ ! -e "$path" ]]; then
      missing+=("$path")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing paper-reproduction prerequisites:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

run_group() {
  local protocol="$1"
  shift
  echo
  echo "==> Running ${protocol} paper protocol ($# configs x ${#seeds[@]} seeds)"
  "${python_cmd[@]}" -m entry_point.run_experiment \
    "${dry_run[@]}" \
    "${devices[@]}" \
    --protocol "$protocol" \
    --seeds "${seeds[@]}" \
    "$@"
}

if [[ ${#dry_run[@]} -eq 0 ]]; then
  ensure_paper_datasets
  preflight
fi

episodic_configs=()
for setting in \
  svhn_to_svhn_single svhn_to_svhn_mixed \
  mnist_to_mnist_single mnist_to_mnist_mixed \
  svhn_to_mnist_single svhn_to_mnist_mixed \
  mnist_to_svhn_single mnist_to_svhn_mixed
do
  for method in \
    baseline adabn tent eata rpt \
    augmix augmix_adabn augmix_eata augmix_tent augmix_rpt
  do
    episodic_configs+=("configs/experiment/table/${setting}/${method}.yaml")
  done
done

for setting in \
  svhn_to_svhn_clean mnist_to_mnist_clean \
  svhn_to_mnist_clean mnist_to_svhn_clean \
  cifar10_to_cifar10_clean cifar10_to_cifar10_1_clean
do
  for method in baseline augmix augmix_adabn augmix_tent augmix_eata augmix_rpt
  do
    episodic_configs+=("configs/experiment/table/${setting}/${method}.yaml")
  done
done

for setting in \
  cifar10_to_cifar10_1_c_single cifar10_to_cifar10_1_c_mixed \
  cifar10_to_cifar10_single cifar10_to_cifar10_mixed
do
  for method in baseline augmix augmix_adabn augmix_eata augmix_tent augmix_rpt
  do
    episodic_configs+=("configs/experiment/table/${setting}/${method}.yaml")
  done
done

continual_short_configs=()
for setting in \
  svhn_to_svhn_mixed mnist_to_mnist_mixed \
  svhn_to_mnist_mixed mnist_to_svhn_mixed
do
  for method in augmix_cotta augmix_eata augmix_tent augmix_rpt augmix_sarpt
  do
    continual_short_configs+=("configs/experiment/table/${setting}/${method}.yaml")
  done
done

continual_long_configs=()
for setting in svhn_to_svhn_mnist_mixed mnist_to_svhn_mnist_mixed
do
  for method in baseline augmix_cotta augmix_eata augmix_tent augmix_rpt augmix_sarpt
  do
    continual_long_configs+=("configs/experiment/table/${setting}/${method}.yaml")
  done
done

run_group episodic "${episodic_configs[@]}"
run_group continual_short "${continual_short_configs[@]}"
run_group continual_long "${continual_long_configs[@]}"

if [[ ${#dry_run[@]} -eq 0 ]]; then
  echo
  echo "==> Building averaged tables and figures"
  "${python_cmd[@]}" -m analysis.analyze_results_two
  "${python_cmd[@]}" -m analysis.analyze_crr
  "${python_cmd[@]}" -m analysis.plot_long_stream_accuracy
fi

echo
echo "==> Paper table reproduction complete"
