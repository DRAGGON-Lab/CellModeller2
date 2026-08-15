# Metal environment

The native Metal backend compiles its independent MSL kernels at runtime
through `MTLDevice`. Validate the complete backend against the CPU reference on
real Apple GPU hardware with:

```console
scripts/run_metal_conformance.sh
```

Every Metal-enabled test build includes a mandatory `metal_runtime_gate`; it
constructs each enumerated native device and therefore compiles every embedded
MSL library. The runner requires a clean worktree, performs a fresh configure
and clean rebuild, executes the complete CTest suite, and records the source
commit, display-device inventory, macOS, Xcode, Clang, Metal framework, logs,
JUnit results, final status, and `SHA256SUMS` for every recorded artifact in a
timestamped directory under `build/`.

The manually dispatched `Metal conformance` workflow runs the same gate on a
self-hosted macOS runner with the custom `metal` label and always uploads its
evidence. It has no pull-request trigger.

Contact geometry uses deterministic sweep-and-prune capsule candidates followed
by native MSL count, inclusive-scan, and fill pipelines.

Rod mechanics uses native MSL Jacobian assembly, matrix-free `B` and `B^T`
applications, vector updates, pairwise reductions, and a host-orchestrated CG
loop. Only scalar reduction results cross back to the host during iteration;
the per-cell vectors remain in Metal buffers until the final correction result.
