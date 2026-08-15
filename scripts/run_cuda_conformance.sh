#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="${1:-${source_dir}/build/cuda-conformance-${timestamp}}"

if [[ -e "${report_dir}" ]]; then
  printf 'report path already exists: %s\n' "${report_dir}" >&2
  exit 2
fi
mkdir -p "${report_dir}"

finish() {
  status=$?
  trap - EXIT
  if [[ ${status} -eq 0 ]]; then
    result=pass
  else
    result=fail
  fi
  {
    printf 'result\t%s\n' "${result}"
    printf 'exit_code\t%s\n' "${status}"
    printf 'completed_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${report_dir}/result.tsv"
  (
    cd "${report_dir}"
    find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | while IFS= read -r file; do
      cmake -E sha256sum "${file}"
    done
  ) >"${report_dir}/SHA256SUMS"
  printf 'CUDA conformance %s; evidence: %s\n' "${result}" "${report_dir}"
  exit "${status}"
}
trap finish EXIT

for command_name in cmake ctest git ninja nvcc nvidia-smi; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'required command is unavailable: %s\n' "${command_name}" >&2
    exit 2
  fi
done

cd "${source_dir}"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'CUDA conformance requires a clean worktree\n' >&2
  git status --short >&2
  exit 2
fi

if ! nvidia-smi -L >"${report_dir}/nvidia-devices.txt"; then
  printf 'nvidia-smi could not discover an NVIDIA device\n' >&2
  exit 2
fi
if [[ ! -s "${report_dir}/nvidia-devices.txt" ]]; then
  printf 'nvidia-smi reported no NVIDIA devices\n' >&2
  exit 2
fi

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'kernel\t%s\n' "$(uname -srvmo)"
  printf 'cmake\t%s\n' "$(cmake --version | head -n 1)"
  printf 'ctest\t%s\n' "$(ctest --version | head -n 1)"
  printf 'nvcc_release\t%s\n' "$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n 1)"
} >"${report_dir}/environment.tsv"

nvidia-smi \
  --query-gpu=index,uuid,name,compute_cap,driver_version,memory.total,pci.bus_id \
  --format=csv,noheader,nounits >"${report_dir}/gpus.csv"
nvcc --version >"${report_dir}/nvcc.txt"
nvidia-smi -q >"${report_dir}/nvidia-smi.txt"

cmake --preset cuda-debug --fresh 2>&1 | tee "${report_dir}/configure.log"
cmake --build --preset cuda-debug --clean-first 2>&1 | tee "${report_dir}/build.log"
ctest --preset cuda-debug --no-tests=error \
  --output-junit "${report_dir}/ctest.xml" 2>&1 | tee "${report_dir}/ctest.log"
