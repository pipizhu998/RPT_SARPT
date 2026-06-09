#!/usr/bin/env bash
set -euo pipefail

FILE_ID="${OUTPUTS_GOOGLE_DRIVE_FILE_ID:-1-0Ax_FKmRNqQJ7z2CqCRdmQXcc20xLiH}"
ARCHIVE_NAME="${OUTPUTS_ARCHIVE_NAME:-outputs.tar.gz}"
DIRECT_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE_PATH="${PROJECT_ROOT}/${ARCHIVE_NAME}"
TMP_DIR="${PROJECT_ROOT}/.tmp/download_outputs"
COOKIE_JAR="${TMP_DIR}/gdrive_cookies.txt"
CONFIRM_PAGE="${TMP_DIR}/gdrive_confirm.html"
MANIFEST="${TMP_DIR}/outputs_manifest.txt"

FORCE_DOWNLOAD=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE_DOWNLOAD=1
elif [[ "${1:-}" != "" ]]; then
  echo "Usage: tools/download_outputs.sh [--force]" >&2
  exit 2
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

archive_is_valid() {
  [[ -s "${ARCHIVE_PATH}" ]] && tar -tzf "${ARCHIVE_PATH}" >/dev/null 2>&1
}

download_outputs_archive() {
  mkdir -p "${TMP_DIR}"
  rm -f "${COOKIE_JAR}" "${CONFIRM_PAGE}" "${ARCHIVE_PATH}.part"

  echo "Downloading Google Drive file ${FILE_ID}"
  curl -L --fail --retry 3 --connect-timeout 20 \
    -c "${COOKIE_JAR}" \
    -o "${CONFIRM_PAGE}" \
    "${DIRECT_URL}"

  if tar -tzf "${CONFIRM_PAGE}" >/dev/null 2>&1; then
    mv "${CONFIRM_PAGE}" "${ARCHIVE_PATH}"
    return
  fi

  local form_action confirm uuid download_url
  form_action="$(
    sed -n 's/.*<form id="download-form" action="\([^"]*\)".*/\1/p' "${CONFIRM_PAGE}" | head -n 1
  )"
  confirm="$(
    sed -n 's/.*name="confirm" value="\([^"]*\)".*/\1/p' "${CONFIRM_PAGE}" | head -n 1
  )"
  uuid="$(
    sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' "${CONFIRM_PAGE}" | head -n 1
  )"

  if [[ -z "${form_action}" ]]; then
    form_action="https://drive.usercontent.google.com/download"
  fi
  if [[ -z "${confirm}" ]]; then
    echo "Could not find the Google Drive confirmation token." >&2
    echo "Open ${CONFIRM_PAGE} to inspect the response from Google Drive." >&2
    exit 1
  fi

  download_url="${form_action}?id=${FILE_ID}&export=download&confirm=${confirm}"
  if [[ -n "${uuid}" ]]; then
    download_url="${download_url}&uuid=${uuid}"
  fi

  curl -L --fail --retry 3 --connect-timeout 20 \
    -b "${COOKIE_JAR}" \
    -c "${COOKIE_JAR}" \
    -o "${ARCHIVE_PATH}.part" \
    "${download_url}"
  mv "${ARCHIVE_PATH}.part" "${ARCHIVE_PATH}"
}

extract_outputs_archive() {
  mkdir -p "${TMP_DIR}"
  tar -tzf "${ARCHIVE_PATH}" > "${MANIFEST}"

  if grep -E '(^/|(^|/)\.\.(/|$))' "${MANIFEST}" >/dev/null; then
    echo "Refusing to extract archive with unsafe paths." >&2
    exit 1
  fi

  local first_member
  first_member="$(sed -n '1p' "${MANIFEST}")"
  if [[ "${first_member}" == "outputs" || "${first_member}" == outputs/* ]]; then
    echo "Extracting ${ARCHIVE_NAME} into ${PROJECT_ROOT}"
    tar -xzf "${ARCHIVE_PATH}" -C "${PROJECT_ROOT}"
  else
    echo "Extracting ${ARCHIVE_NAME} into ${PROJECT_ROOT}/outputs"
    mkdir -p "${PROJECT_ROOT}/outputs"
    tar -xzf "${ARCHIVE_PATH}" -C "${PROJECT_ROOT}/outputs"
  fi
}

require_command curl
require_command tar
require_command sed
require_command grep

if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && archive_is_valid; then
  echo "Using existing archive: ${ARCHIVE_PATH}"
else
  download_outputs_archive
fi

if ! archive_is_valid; then
  echo "Downloaded file is not a valid gzip tar archive: ${ARCHIVE_PATH}" >&2
  exit 1
fi

extract_outputs_archive
echo "Done. Outputs are under ${PROJECT_ROOT}/outputs"
