# CellModeller2 tutorials

This tutorial set ports the maintained CellModeller wiki lessons and the
CellModeller-oriented SimBOL examples to the native CellModeller2 modeling
interface. It is intended to be read in order, but every executable model is
self-contained and can be used independently.

## Start here

1. [Run, inspect, and resume a model](getting-started.md)
2. [Growth, division, cell types, and constraints](biophysics-and-growth.md)
3. [Intracellular species and gene circuits](intracellular-dynamics.md)
4. [Diffusible signals and cell-cell communication](signaling.md)
5. [Plasmid segregation, contacts, and conjugation](discrete-state-and-contacts.md)
6. [Checkpoints, contact graphs, and quantitative analysis](analysis.md)
7. [SimBOL circuit ports](simbol.md)

The tutorials use `uv`, the `cm` command, data-only checkpoints, and the
standalone viewer. They do not require OpenCL, injected C source, executable
pickle files, or the legacy GUI.

## Source and coverage ledger

The legacy Python files are pinned at CellModeller commit
`4896f543c6250f053eea2312e628cc3a96bf7408`, matching the executable
compatibility matrix. The prose inventory uses CellModeller wiki commit
`95587c11899677b7cba87c64bc20210fb1f8f6ce`. The SimBOL inventory uses commit
`54501f9da6f9809588be48b854a6c4f8abd933b5`.

| Source lesson | Native tutorial coverage | Status |
| --- | --- | --- |
| Wiki Tutorial 1a | `biophysics.py`, `basics` | native, restartable |
| Wiki Tutorial 1b | `biophysics.py`, `competition` | native, restartable |
| Wiki Tutorial 1c | `biophysics.py`, `box` | native, restartable |
| Wiki Tutorial 2a | `gene_expression.py`, `constitutive` | native typed rates |
| Wiki Tutorial 2b | `gene_expression.py`, `oscillator` | native typed rates |
| Wiki Tutorial 3 | `signaling.py`, `mutualism` | native coupled rates |
| Old Example 1 and its two exercises | `biophysics.py`, `basics`, `two_types`, `short_cells` | native, restartable |
| Old Example 2 | `gene_expression.py`, `legacy_constitutive`, `dilution`, `derepression` | native typed rates |
| Old Example 3 | `signaling.py`, `single_gene` | native coupled rates and planes |
| Old Example 4 | `signaling.py`, `communication` | native coupled rates and planes |
| Old Example 5 | `plasmid_segregation.py` | native discrete controller |
| Contact graph and live conjugation | `conjugation.py` plus the analysis tutorial | native contact API |
| Analyzing Data | analysis tutorial and the existing recipe library | Parquet/Zarr workflow |
| SimBOL `CM_BBa_01`–`05`, `CM_BBa_I5200` | `simbol_circuits.py` | native typed rates |
| SimBOL `CM_Danino.py` | `danino_clock.py` | core clock and trap port; bounded transport difference documented |
| SimBOL CellModeller notebook | SimBOL tutorial | workflow documented; the checked-in notebook ends before generation |

The compatibility copies under `examples/legacy` remain the exact equation
migrations used by the legacy matrix. The files under `examples/tutorials`
are teaching models: they consolidate related lessons, expose scenarios as
parameters, use current names and units, and implement exact resume.

## What “ported” means

The ports preserve the biological question, initial conditions, rate
equations, strain roles, division rule, and physical geometry where those are
well-defined. They deliberately express them through current contracts:

- `CellInit` defines native cell state;
- `NativeController` or a small explicit controller owns regulation and
  stochastic state;
- `RatePlanBuilder` replaces runtime OpenCL source injection;
- `SignalGridSpec` and a `CoupledRatePlan` jointly define transport and
  cell-grid exchange;
- plane and sphere constraints replace encoded obstacle rows;
- checkpoints replace executable pickle snapshots; and
- scene and analysis exports replace model-owned renderer and analysis code.

Several old sources mix “length” and `targetVol`, use callback attributes as
both molecule counts and concentrations, or perform chemical updates once per
GUI step without multiplying by `dt`. Each tutorial calls out the chosen
CellModeller2 interpretation instead of silently preserving an ambiguity.
