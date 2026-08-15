# Legacy rod-mechanics audit

This audit separates the biological and numerical behavior worth preserving
from the OpenCL-era storage and scheduling choices that should not cross into
CellModeller2.

## Sources and authority

The legacy mechanics path is split across three sources:

- `CLBacterium.py` owns configuration, host/device storage, broad-phase setup,
  substep scheduling, and conjugate-gradient iteration.
- `CLBacterium.cl` owns capsule geometry, constraint generation, matrix-free
  operator applications, displacement updates, and integration.
- `Doc/BacterialBiophysics/BacterialBiophysics.tex` explains the original
  impulse formulation, but it does not exactly describe the code that runs.

For compatibility, defined behavior in the Python and OpenCL implementation is
the baseline. The TeX document is explanatory evidence, not an executable
specification. Undefined behavior, unchecked capacity, and internally
inconsistent behavior are defects to repair rather than contracts to preserve.

## Observed discrete model

A cell is a capsule whose centerline is a segment with center `x`, unit axis
`a`, cylindrical length `l`, and radius `r`. Growth supplies a desired length
increment. Mechanics solves a seven-degree-of-freedom correction per cell:

```text
delta_q = [delta_x, delta_theta, delta_l]
```

The narrow phase computes closest points `p_a` and `p_b` on two centerlines.
Its signed surface separation is

```text
d = norm(p_b - p_a) - (r_a + r_b)
```

so negative `d` means penetration. A cell-cell row is activated when
`d < 0.01`. Non-parallel rods create one row. Near-parallel rods with
overlapping projected intervals create two rows, each weighted by `1/sqrt(2)`.
The squared weights make the pair's total least-squares weight equal to that of
a one-row contact.

For a contact normal `n` pointing from cell `a` toward cell `b`, the implemented
row entries are equivalent to

```text
J_a = [n, r_a_contact cross n,
       dot(a_axis, r_a_contact) * dot(a_axis, n) / (l_a + 2 r_a)]
J_b = [n, r_b_contact cross n,
       dot(b_axis, r_b_contact) * dot(b_axis, n) / (l_b + 2 r_b)]
B_k = w_k [J_a, -J_b]
b_k = w_k d_k
```

where each contact arm is measured from the cell center. Plane and sphere rows
have only the cell-side entry and multiply `w_k` by their configured
coefficient.

The implementation solves the regularized normal equations

```text
(B transpose B + gamma^-1 M) delta_q = B transpose b
```

with unpreconditioned conjugate gradient. `M` weights translation by
`muA * (l + 2r)`, rotation by a slender-rod inertia using total capsule length,
and length by `gamma`. The default stopping test is RMS residual below `5e-3`,
with at most `7 * cell_count` iterations. Angular correction is clamped to five
degrees per integration and mechanics may reduce desired elongation to zero,
but does not directly shorten a cell.

Despite names such as `CGSSolve`, `add_impulse`, and the collision language in
the TeX document, the running code is best understood as a regularized
position-and-growth correction, not a time-accurate momentum impulse. The
`dt` and substep `alpha` arguments do not affect the assembled operator.

## Legacy representation

Contacts are stored in a fixed `cell_count * max_contacts` table, owned by the
lower slot of a cell pair. A second fixed table collects reverse adjacency.
Plane and sphere endpoints are encoded as negative integers. A two-dimensional
uniform grid over `x` and `y` supplies cell candidates; the narrow phase still
uses three-dimensional vectors.

The following are storage artifacts, not scientific semantics:

- a default hard limit of 24 outgoing and 24 incoming contacts per cell;
- host-side sorting and grid-bound discovery each step;
- negative endpoint identifiers for different constraint kinds;
- zero-filled inactive rows required by the dense-stride layout;
- OpenCL `float4` and `float8` packing.

## Defects and ambiguities

CellModeller2 must not reproduce these behaviors:

1. Contact insertion does not bounds-check `max_contacts`, so dense colonies can
   write past a cell's contact row.
2. Plane and sphere contacts reuse the same negative endpoint encoding and can
   be mistaken for one another.
3. Exact coincident centerlines normalize a zero vector, producing an undefined
   contact normal.
4. The broad phase assumes a fixed grid spacing is sufficient for every rod and
   only indexes two spatial dimensions.
5. Conjugate gradient does not diagnose zero or non-finite `p transpose A p`.
6. The documented finite-radius cylinder inertia differs from the slender-rod
   tensor used by the kernel, whose axial rotational penalty is zero.
7. Contact history is retained during a substep search but cleared at the start
   of each outer step, making the intended hysteresis unclear.
8. `ct_overlap` is written but never consumed by mechanics.
9. Solver parameters named `dt` and `alpha` are passed through but unused.

CellModeller2 resolves item 6 with a finite-radius, full-rank rotational inertia
tensor. This is a deliberate departure from numerically defined legacy output,
not an accidental consequence of a backend port.

Correction integration retains the five-degree angular cap and the
`max(0, desired elongation + length correction)` non-shortening rule. It makes
solver convergence a checked precondition and validates the entire update
before mutating geometry, neither of which the legacy path enforced.

These differences require explicit fixtures before any claim of legacy parity.
They are not grounds for silently adopting an accidental result.

## Required compatibility fixtures

The mechanics implementation needs source-controlled fixtures for:

- separated, end-on, skew, parallel, anti-parallel, and collinear capsule pairs;
- exact coincident centerlines with a deterministic finite normal;
- pair-order symmetry and stable-ID ordering;
- more than 24 contacts incident on one cell;
- plane and inside/outside sphere constraints;
- matrix-free operator symmetry and positive definiteness after regularization;
- solver convergence, breakdown reporting, and residual recomputation;
- a small growing colony compared with a recorded legacy trajectory where the
  legacy result is numerically defined.

The legacy trajectory is evidence for compatibility. Analytic geometry and
operator invariants remain the authority when the legacy implementation is
undefined.
