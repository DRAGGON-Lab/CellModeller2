# CUDA environment

The native CUDA growth, contact-geometry, mechanics, signaling, and
coupled-rate slices use CUDA C++ kernels and the CUDA Runtime API directly.
Contact generation has native count, inclusive-scan, and fill kernels.
Mechanics has native Jacobian assembly, matrix-free operator, vector update,
and pairwise reduction kernels with a host-orchestrated CG loop. Coupled rates
use ordered cell and grid kernels with device-resident sampling,
interpretation, transport, and deterministic source gathering. Configure,
build, and test them on a machine with the CUDA Toolkit and an NVIDIA GPU:

```console
scripts/run_cuda_conformance.sh
```

`CM2_ENABLE_CUDA` is off by default so a CPU build never acquires an accidental
CUDA toolchain dependency. Every CUDA-enabled test build includes a mandatory
`cuda_runtime_gate`; it fails when the runtime cannot discover and construct
the native backend. The runner requires a clean worktree, performs a fresh
configure and clean rebuild, executes the complete CTest suite, and writes a
timestamped evidence directory under `build/` with the source commit, device
inventory, driver and toolkit details, logs, JUnit results, an explicit
pass/fail record, and `SHA256SUMS` for every recorded artifact. Pass a path as
its sole argument to choose a different new evidence directory.

The manually dispatched `CUDA conformance` GitHub Actions workflow runs this
same gate on a self-hosted runner carrying the default `self-hosted`, `linux`,
and `x64` labels plus the custom `gpu` label. It is intentionally not triggered
by pull requests, so untrusted branch code is not sent automatically to a
persistent GPU host. Keep the Actions runner current enough for
`actions/checkout@v6` and `actions/upload-artifact@v7`; the workflow always
uploads the evidence directory, including a failed gate's `result.tsv`.

The CUDA 12.8 toolkit compiles and links the full CUDA-enabled project in
NVIDIA's official development container. Reproduce that driverless gate with:

```console
scripts/run_cuda_compile_check.sh
```

The script mounts the source read-only, configures and links all native CUDA
targets inside an ephemeral CUDA 12.8.1 container, lists the registered tests
without executing them, and records the exact container image plus checksummed
logs. Override `CM2_CUDA_ARCHITECTURES` or `CM2_CUDA_CONTAINER_IMAGE` when a
different compile target is required. This proves source and link compatibility
only; it does not establish CUDA conformance. Every CUDA hardware gate remains
pending until the shared tests are recorded on an NVIDIA device. Hardware test
reports must include the device, driver, toolkit, and target compute capability.

Contact generation uses deterministic sweep-and-prune capsule-bound staging,
then native CUDA count, scan, and fill kernels for the narrow phase.
