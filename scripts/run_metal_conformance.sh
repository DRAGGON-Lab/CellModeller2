#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="${1:-${source_dir}/build/metal-conformance-${timestamp}}"

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
  printf 'Metal conformance %s; evidence: %s\n' "${result}" "${report_dir}"
  exit "${status}"
}
trap finish EXIT

for command_name in clang cmake ctest defaults git ninja sw_vers system_profiler xcodebuild; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'required command is unavailable: %s\n' "${command_name}" >&2
    exit 2
  fi
done

if [[ "$(uname -s)" != Darwin ]]; then
  printf 'Metal conformance requires macOS\n' >&2
  exit 2
fi

cd "${source_dir}"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Metal conformance requires a clean worktree\n' >&2
  git status --short >&2
  exit 2
fi

{
  printf 'source_commit\t%s\n' "$(git rev-parse HEAD)"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host\t%s\n' "$(hostname)"
  printf 'kernel\t%s\n' "$(uname -a)"
  printf 'cmake\t%s\n' "$(cmake --version | head -n 1)"
  printf 'ctest\t%s\n' "$(ctest --version | head -n 1)"
  printf 'metal_framework\t%s\n' "$(defaults read /System/Library/Frameworks/Metal.framework/Versions/A/Resources/Info CFBundleShortVersionString)"
} >"${report_dir}/environment.tsv"

sw_vers >"${report_dir}/macos.txt"
xcodebuild -version >"${report_dir}/xcode.txt"
clang --version >"${report_dir}/clang.txt"
system_profiler SPDisplaysDataType -json >"${report_dir}/displays.json"

cmake --preset metal-debug 2>&1 | tee "${report_dir}/configure.log"
cmake --build --preset metal-debug 2>&1 | tee "${report_dir}/build.log"
ctest --preset metal-debug --no-tests=error \
  --output-junit "${report_dir}/ctest.xml" 2>&1 | tee "${report_dir}/ctest.log"
