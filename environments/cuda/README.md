# CUDA environment

The native CUDA growth, contact-geometry, and mechanics slices use CUDA C++
kernels and the CUDA Runtime API directly. Contact generation has native count,
inclusive-scan, and fill kernels. Mechanics has native Jacobian assembly,
matrix-free operator, vector update, and pairwise reduction kernels with a
host-orchestrated CG loop. Configure, build, and test them on a machine with the
CUDA Toolkit and an NVIDIA GPU:

```console
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

`CM2_ENABLE_CUDA` is off by default so a CPU build never acquires an accidental
CUDA toolchain dependency. The conformance test constructs a real CUDA backend
and cannot pass through a CPU fallback.

The CUDA 12.8 toolkit compiles and links the full CUDA-enabled project in
NVIDIA's official arm64 development container. That driverless validation does
not establish CUDA conformance. Growth, contact, and mechanics conformance
remain pending until the tests are recorded on NVIDIA hardware. Test reports
must include the device, driver, toolkit, and target compute capability.

The current contact implementation uses exhaustive pair staging to establish
geometry correctness. The scalable capsule-bounds broad phase remains required
before contact generation is production-ready.
