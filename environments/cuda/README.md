# CUDA environment

The native CUDA growth slice uses CUDA C++ kernels and the CUDA Runtime API
directly. Configure, build, and test it on a machine with the CUDA Toolkit and
an NVIDIA GPU:

```console
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug
```

`CM2_ENABLE_CUDA` is off by default so a CPU build never acquires an accidental
CUDA toolchain dependency. The conformance test constructs a real CUDA backend
and cannot pass through a CPU fallback.

The source is present, but CUDA conformance remains pending until this test is
recorded on NVIDIA hardware. Test reports must include the device, driver,
toolkit, and target compute capability.
