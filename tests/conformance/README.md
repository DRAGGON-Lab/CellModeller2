# Growth conformance scenario

The growth scenario is compiled into every test-enabled build and runs once
for each backend present in that build. It deliberately uses 513 heterogeneous
cells so native launches cross common threadgroup and block boundaries.

The scenario checks the declared explicit Euler recurrence independently:

```text
length[n + 1] = length[n] + growth_rate * length[n] * dt
```

Lengths use an absolute tolerance of `1e-6` and a relative tolerance of `1e-6`.
Stable identifiers, compact slots, growth rates, cell types, backend identity,
and structural validation are exact checks.

A backend only earns conformance when this executable passes with that native
backend enabled on its real hardware. A CPU-only run validates the scenario and
reference implementation, but says nothing about GPU conformance.

## Lifecycle conformance scenario

The lifecycle scenario interleaves growth with two generations of deterministic
equal division. Stable identifiers, slot reuse, active ordering, lineage,
daughter geometry, inherited cell attributes, and simulation time are checked.
Floating-point length and position checks use the same `1e-6` absolute and
relative tolerances as the growth scenario; lifecycle identities are exact.

## Contact conformance scenario

The contact scenario compares each contact-capable backend with the exhaustive
CPU oracle for mixed end-on, parallel, anti-parallel, skew, point-like, empty,
and single-cell geometry. It also checks stable-ID ordering after slot reuse and
a 31-cell coincident case with 60 incident contacts per cell. Contact fields use
absolute and relative tolerances of `2e-5`; IDs, slots, ordinals, graph sizes,
and incidence indices are exact.

## Mechanics operator fixtures

The CPU mechanics fixtures probe the declared regularizer, operator symmetry
and positive definiteness, solver convergence, exact residual recomputation,
iteration-limit reporting, and non-finite-curvature breakdown. They are unit
fixtures rather than cross-backend conformance: a native mechanics row will be
added only when Metal or CUDA implements the complete operator and solver loop.
