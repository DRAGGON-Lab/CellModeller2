# ADR 0002: typed dynamic contact mechanics

- Status: accepted
- Date: 2026-08-14

## Context

Rod contact and mechanical relaxation are the first CellModeller feature that
requires irregular GPU work, dynamic topology, reductions, and iterative
solves. The legacy fixed-stride contact table is unsafe at high density and
mixes cell pairs, planes, and spheres through integer conventions. A generic
portable GPU abstraction would also prevent Metal and CUDA from using their
native scan, sort, reduction, and synchronization facilities.

## Decision

CellModeller2 represents a contact as a typed record with stable cell IDs,
current compact slots, contact ordinal, surface point, unit normal, signed
separation, and row weight. External constraints use tagged plane or sphere
references rather than sentinel cell IDs.

The CPU reference performs an exhaustive pair search. It is intentionally
simple and is the geometry oracle for small conformance scenarios.

Native GPU broad and narrow phases use a two-pass dynamic pipeline:

1. conservatively generate candidate pairs from capsule bounds;
2. count zero, one, or two narrow-phase contacts per pair;
3. exclusive-scan the counts into offsets;
4. allocate or grow the contact arrays;
5. fill the typed contact records.

There is no scientific contact cap. Allocation failure is explicit. In
deterministic mode, contacts are ordered by stable endpoint IDs and contact
ordinal. Fast mode may use backend-native ordering, but consumers must not infer
identity from contact-array position.

The narrow phase preserves the legacy one-row/two-row distinction and the
`1/sqrt(2)` two-row weighting. Degenerate normals are resolved deterministically
by projecting center displacement perpendicular to the rod axis; if that also
vanishes, the least-aligned Cartesian basis axis defines a perpendicular. Every
emitted normal must be finite and unit length.

The solver exposes a matrix-free operator rather than a materialized sparse
matrix. The initial compatibility operator is the regularized seven-DOF system
recorded in the legacy audit. The public result includes convergence status,
iteration count, initial and final residuals, and a breakdown reason. A backend
may use native reductions and preconditioning, but it must apply the same
declared operator and convergence criterion.

CPU, Metal, and CUDA own separate implementations. They share contact fixtures,
operator probes, tolerances, and exact endpoint invariants. Metal uses Metal
compute pipelines and MSL; CUDA uses CUDA C++ and the CUDA Runtime API.

## Initial delivery order

1. Robust capsule geometry and exhaustive CPU contact generation.
2. Shared contact fixtures and a dynamic contact-graph API.
3. Native Metal and CUDA count/scan/fill pipelines.
4. CPU matrix-free operator and diagnosed conjugate-gradient solver.
5. Native Metal and CUDA operator, reductions, and solver loops.
6. Typed plane and sphere constraints.

This order isolates geometry disagreements before solver behavior can hide
them. A feature ledger entry advances only when the corresponding shared
fixture passes on real backend hardware.

## Consequences

- Contact storage scales with actual topology rather than a per-cell maximum.
- Stable cell identity survives compaction while kernels continue to use dense
  slots.
- GPU implementations duplicate orchestration where native primitives differ.
- Deterministic mode may require an ordering pass that fast mode can skip.
- Legacy trajectories with undefined normals or overflowed contact rows are not
  reproducibility targets.
