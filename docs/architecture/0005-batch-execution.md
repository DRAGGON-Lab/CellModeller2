# ADR 0005: deterministic batch execution

- Status: accepted
- Date: 2026-08-15

## Context

CellModeller2 needs one reproducible, non-interactive path for local runs,
cluster jobs, backend conformance, and future workflow orchestration. Legacy
batch scripts assemble simulator state through ambient module globals and mix
model construction, output policy, and execution. They do not give every run
an explicit backend, device, seed, or collision policy.

Checkpoint loading must also remain distinct from Python model execution. A
checkpoint is portable data and safe to parse; a model file is trusted Python
code chosen explicitly by the operator.

## Decision

The `cm2` console command has two initial operations:

- `cm2 devices` enumerates native CPU, Metal, and CUDA devices; and
- `cm2 run` either executes a Python file supplied with `--model` or restores a
  data-only checkpoint supplied with `--resume`.

A Python model exports `build(context)` and returns a `Simulation` created by
`context.simulation()`. The context owns the selected backend and device, an
unsigned 64-bit seed, a dedicated Python pseudorandom generator, and immutable
JSON parameters. The runner rejects a returned simulation on another backend
or device. Models remain ordinary Python and may define their own typed helper
layers without coupling the engine to a particular modeling DSL.

The run loop uses an integer step count and an explicit finite non-negative
time step. Before simulation begins, the runner checks the final and all
periodic checkpoint paths for collisions. It does not overwrite existing data
unless `--overwrite` is present. Every write uses the checkpoint layer's
atomic-replace behavior. Periodic filenames contain an eight-digit completed
step number, and the final checkpoint is written even for a zero-step run.

Run provenance records the model or resume input's absolute path and SHA-256
digest. Model provenance also records the seed and JSON parameters. Each
checkpoint records the requested and completed steps, time step, and run
status. Randomness during model construction is reproducible when the model
uses `context.rng`; runtime stochastic mechanisms will receive explicit engine
random streams in a later feature slice.

## Consequences

- Batch jobs have one inspectable entry point and stable exit behavior.
- Backend and device selection cannot be silently overridden by a model.
- Identical source, seed, parameters, backend, and numeric environment provide
  a reproducible construction and execution contract.
- Model files are trusted executable code and must not be confused with safe
  checkpoint input.
- Scheduling systems can treat periodic checkpoints as independent restart
  points without parsing process logs.
