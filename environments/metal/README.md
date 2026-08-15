# Metal environment

The first native Metal vertical slice compiles its MSL growth kernel at runtime
through `MTLDevice` and validates it against the CPU reference on real Apple GPU
hardware. Run it with:

```console
cmake --preset metal-debug
cmake --build --preset metal-debug
ctest --preset metal-debug
```

Metal test reports should record the device, OS, Xcode, and shader compiler
versions. Ahead-of-time metallib packaging will be added when the static kernel
set begins to stabilize.

The contact geometry slice also runs native MSL count, inclusive-scan, and fill
pipelines. Its current exhaustive pair staging is a correctness implementation,
not the production broad phase; the feature ledger keeps that distinction
explicit.
