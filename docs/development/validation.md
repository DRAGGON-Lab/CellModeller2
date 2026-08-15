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
orientation for inside and outside spheres. Metal now runs the shared geometry
and mechanics fixtures on Apple GPU hardware. Until the CUDA port lands,
adding a constraint to a CUDA simulation makes mechanics fail explicitly.

A CUDA toolkit-only build proves source and link compatibility, not backend
conformance. CUDA rows advance only after the shared executable runs on an
NVIDIA device. `backend_available` reports runtime device availability, while
`Simulation.supports` reports features implemented by a constructed backend.

Pull requests must not claim a backend supports a feature when it invokes the
CPU reference or transfers the full state to the host to complete the step.
