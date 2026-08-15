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

Batch fixtures execute a real model file through both the Python API and the
`cm2` entry point. They verify seeded construction, JSON parameters, periodic
checkpoint names, collision preflight, model and resume hashes, exact resumed
time, and machine-readable device discovery. The same model and seed must
produce the same serialized simulation state.

Signal-grid CPU fixtures cover mass-conserving no-flux diffusion, fixed
reservoir values, periodic upwind advection, reduced-dimensional and 3D
trilinear samples, paired-periodic validation, explicit stability rejection,
checkpoint round trips, and v1/v2-to-v3 checkpoint migration. The shared
anisotropic 630-value scenario executes native MSL transport on Apple GPU
hardware and compares it with the CPU reference at `5e-6`. The independent
CUDA transport kernel compiles and links against the CUDA 12.8 toolkit, but
compilation and driverless test execution are not conformance; the same gate
must execute on NVIDIA hardware.

Coupled-rate CPU fixtures verify that sampling and scatter use the same
trilinear weights, cell signal outputs are amount-per-time divided by voxel
volume, transport samples the old field, species rates see post-growth diluted
concentrations, invalid cell positions fail before growth, and the complete
plan survives a v3 checkpoint round trip. Checkpoint tests separately verify
that v1 and v2 files migrate with no coupled plan. A backend must execute the
same coupled stage without a host fallback before advertising coupled-rate
support.
The shared fixture uses 513 heterogeneous cells, two signals, three species,
anisotropic transport, fractional samples, and repeated scatter destinations;
native implementations compare every committed value with the CPU reference.
The native Metal stage passes that fixture on Apple GPU hardware. It evaluates
cells and gathers grid sources in two MSL kernels within one command buffer;
the grid-thread gather is deterministic and avoids a floating-point atomic
requirement while keeping intermediate samples and rates device-resident.

A CUDA toolkit-only build proves source and link compatibility, not backend
conformance. CUDA rows advance only after the shared executable runs on an
NVIDIA device. `backend_available` reports runtime device availability, while
`Simulation.supports` reports features implemented by a constructed backend.

Pull requests must not claim a backend supports a feature when it invokes the
CPU reference or transfers the full state to the host to complete the step.
