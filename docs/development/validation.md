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

SBML fixtures parse Level 3 Version 2 XML through libSBML, compile ordered
species metadata, local and global parameters, stoichiometry, arithmetic,
power, exponential, and logarithmic MathML into `SpeciesRatePlan`, and execute
the result through every available native species backend. Rejection fixtures
cover malformed input, non-unit compartments, rules, unsupported MathML, and
unresolved valid SBML symbols. These tests validate the semantic compiler;
the existing 513-cell species fixture remains the scaling and native-device
interpreter gate.

Scene fixtures capture the same rod, lineage, species, and signal-grid state
from every available backend and compare the presentation semantics after
normalizing the expected backend identity fields. Format tests cover exact
round trips, atomic writes, SHA-256 tamper detection, closed schemas, malformed
input, float32 geometry invariants, grid cardinality, and unsigned 64-bit IDs
beyond JavaScript's exact integer range.

The independent viewer validates with `pnpm --dir viewer format:check`,
`check`, `test`, and `build`. Its unit suite verifies a Python-authored RFC 8785
digest, strict browser decoding, tamper rejection, browser-safe cell IDs,
categorical and scalar color mappings, and every signal-slice indexing path.
Rendered browser QA loads `examples/viewer_scene.py` output and exercises rod
picking, selection clearing, species coloring, and signal-axis changes while
checking the console for warnings and errors.

Live-viewer fixtures run a real model through the same resettable factory used
by `cm2 view`. They test stepping, deterministic rebuild, atomic checkpoint
output, strict command parsing, same-origin and bearer-token rejection, and a
real loopback WebSocket exchange. Browser protocol tests independently verify
the embedded scene digest and reject unknown or tampered messages. Rendered
live QA additionally exercises play, pause, step, reset, and checkpoint while
confirming that camera and presentation state remain browser-owned.

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
checkpoint round trips, v1-through-v5 migration, and diagnosed Crank-Nicolson
convergence. The shared anisotropic 630-value scenario executes both Forward
Euler and Crank-Nicolson through native MSL transport on Apple GPU hardware and
compares them with the CPU reference at `5e-6`. The independent
CUDA Forward Euler and Crank-Nicolson kernels compile and link against the CUDA
12.8 toolkit, but compilation and driverless test execution are not
conformance; the same gate must execute on NVIDIA hardware.

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
The native Metal stage passes that fixture in both Forward Euler and
Crank-Nicolson modes on Apple GPU hardware. It evaluates cells and gathers grid
sources in two MSL kernels within one command buffer;
the grid-thread gather is deterministic and avoids a floating-point atomic
requirement while keeping intermediate samples and rates device-resident.
The independent CUDA C++ stage uses the same stream ordering, adding native
Jacobi and residual-reduction kernels for Crank-Nicolson, and compiles and links
for `sm_75` with the CUDA 12.8.1 toolkit. Driverless tests
exercise only the CPU paths because no CUDA device can be constructed; the
513-cell fixture, empty-colony path, and failed-source atomicity check remain
mandatory on NVIDIA hardware before CUDA conformance is claimed.

A CUDA toolkit-only build proves source and link compatibility, not backend
conformance. CUDA rows advance only after the shared executable runs on an
NVIDIA device. `backend_available` reports runtime device availability, while
`Simulation.supports` reports features implemented by a constructed backend.

Pull requests must not claim a backend supports a feature when it invokes the
CPU reference or transfers the full state to the host to complete the step.

Fixed-cell fixtures verify stable-ID mutation, division inheritance,
checkpoint v6 persistence, migration of v1-through-v5 cells to movable state,
the projected mechanics operator and right-hand side, and integration that
rejects contact correction while retaining declared growth. The shared
relaxation scenario passes on CPU and native Metal, including exact zero
correction and unchanged geometry for the fixed cell. CUDA hardware execution
remains required before CUDA fixed-cell support is claimed; the independent
CUDA mask upload and projected mechanics kernels compile and link with the
CUDA 12.8.1 toolkit.

Analysis-export fixtures read the published artifacts back through PyArrow and
Zarr rather than trusting writer completion. They verify explicit Arrow types,
stable and parent identities, cylinder versus full-capsule length, long-form
species rows, complete typed cell and constraint contacts, source and
reconstruction backend provenance, five-dimensional signal order, signal
epoch boundaries, source-path privacy, time ordering, and non-destructive
output collision handling. CUDA contact derivation stays rejected until the
NVIDIA hardware conformance gate passes; ordinary CUDA checkpoint state can
still be exported without a contact derivation.

Recipe fixtures reopen a digest-verified dataset and evaluate the lazy Polars
plans. They cover exact radial and length bin boundaries, retained empty bins,
null means for absent species channels, full-capsule length weighting, contact
row collapse, null-safe sister lineage, and invalid-edge rejection. Signal
fixtures address arrays by the declared `(frame, channel, x, y, z)` order and
verify both a named 2D plane and one voxel's physical-time course. Separate
tamper cases alter a Parquet file and manifest options and must fail before a
recipe reads scientific values.
