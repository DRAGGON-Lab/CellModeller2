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
the native backend. The runner requires a clean worktree, executes the complete
CTest suite, and writes a timestamped evidence directory under `build/` with
the source commit, device inventory, driver and toolkit details, logs, JUnit
results, and an explicit pass/fail record. Pass a path as its sole argument to
choose a different new evidence directory.

The CUDA 12.8 toolkit compiles and links the full CUDA-enabled project in
NVIDIA's official arm64 development container. That driverless validation does
not establish CUDA conformance. Every CUDA hardware gate remains pending until
the shared tests are recorded on an NVIDIA device. Test reports must include
the device, driver, toolkit, and target compute capability.

Contact generation uses deterministic sweep-and-prune capsule-bound staging,
then native CUDA count, scan, and fill kernels for the narrow phase.
