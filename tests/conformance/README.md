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
