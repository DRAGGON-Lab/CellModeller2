# Validation workflow

The project uses vertical conformance slices. For every feature:

1. State the governing equation and discrete algorithm.
2. Identify whether legacy behavior is authoritative, merely informative, or
   an apparent defect.
3. Add focused CPU unit tests.
4. Add a small scenario fixture with declared tolerances.
5. Implement Metal and CUDA independently.
6. Run the same fixture on real Apple and NVIDIA hardware.
7. Add scaling and long-run statistical tests only after stage-level parity.

The first shared scenario is `growth_conformance`. Every test-enabled build
runs the same fixture against each backend compiled into that build. Its exact
state checks and numerical tolerances are recorded beside the test in
`tests/conformance/README.md`.

External constraint geometry begins with focused CPU fixtures covering plane
normal normalization, one- and two-endpoint weighting, inside/outside sphere
orientation, stable typed IDs, and deterministic sphere-center degeneracy.
These fixtures become shared backend conformance tests when the native Metal
and CUDA geometry pipelines are added.

CPU mechanics fixtures additionally verify the external-row operator and
right-hand side directly, solver convergence for a plane, and correction
orientation for inside and outside spheres. Metal runs the shared geometry and
mechanics fixtures on Apple GPU hardware. CUDA has independent native geometry
and mechanics implementations; NVIDIA hardware execution remains its
conformance gate.

Species fixtures validate fixed schema, zero initialization, division
inheritance, plan topology, legacy-compatible effective-volume dilution,
post-dilution rate evaluation, and simultaneous Euler updates. The shared
513-cell scenario crosses common GPU launch boundaries and compares native
state with the CPU reference after heterogeneous time steps. A backend does not
advertise species support until that scenario executes on its hardware.
The native MSL interpreter passes this gate on Apple GPU hardware. The
independent CUDA C++ interpreter compiles and links against the CUDA 12.8
toolkit; execution of this fixture on NVIDIA hardware remains its conformance
gate.

Checkpoint fixtures compare every persisted field exactly before taking a
resumed step. They then continue the same typed species model on each available
backend and compare with a fresh CPU restore under the species tolerance. File
tests cover atomic replacement, provenance, version rejection, duplicate and
unknown structure, finite-number bounds, SHA-256 corruption detection, and
native state validation. Passing on CPU establishes exact host restoration;
passing with Metal or CUDA enabled additionally establishes backend-independent
reconstruction and continued execution on that hardware.

A CUDA toolkit-only build proves source and link compatibility, not backend
conformance. CUDA rows advance only after the shared executable runs on an
NVIDIA device. `backend_available` reports runtime device availability, while
`Simulation.supports` reports features implemented by a constructed backend.

Pull requests must not claim a backend supports a feature when it invokes the
CPU reference or transfers the full state to the host to complete the step.
