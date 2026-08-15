# SimBOL circuit ports

SimBOL connects an SBOL 3 design to simulator-specific code through a
summarized JSON representation. Its CellModeller plugin detects circuit
topology, combines it with numerical parameters, and emits a legacy
CellModeller callback script. This tutorial ports the six checked-in BioBrick
outputs and the Danino clock example to typed CellModeller2 models.

These files are reviewed, source-grounded example ports. They are not a claim
that CellModeller2 currently contains a general SBOL-to-rate-plan compiler.
Topology extraction remains in SimBOL; a future generic integration should
define and version the intermediate JSON schema, parameter units, unsupported
SBOL semantics, and provenance before accepting arbitrary designs.

## SimBOL workflow inventory

At SimBOL commit `54501f9da6f9809588be48b854a6c4f8abd933b5`, the intended flow is:

```text
SBOL 3 document
  -> sbol3_to_json_converter
  -> summarized circuit JSON
  -> parameter preparation / UI
  -> CellModeller-specific generated Python
```

The checked-in `notebooks/CellModeller.ipynb` contains four code cells:
environment setup, SBOL upload, SBOL-to-JSON conversion, and parameter-form
display. It ends before calling `generate_script` or downloading an output.
The generated `test/CM_*.py` files and their matching clean JSON fixtures are
therefore the concrete tutorial outputs used for this port.

## Run the six circuits

Use one model and select a circuit:

```console
uv run cm view \
  --model examples/tutorials/simbol_circuits.py \
  --parameter circuit='"bba_0001"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

Allowed names are `bba_0001`, `bba_0002`, `bba_0003`, `bba_0004`,
`bba_0005`, and `bba_i5200`. Choose `Species` coloring and the reporter channel
listed below.

| Circuit | Ordered species channels | Reporter channel | Additional input |
| --- | --- | ---: | --- |
| BBa_0001 | GFP | 0 | none |
| BBa_0002 | RFP, TetR | 0 | `inducer_concentration`, default 0 aTc |
| BBa_0003 | LuxR, GFP, LuxI, LuxR–AHL pool | 1 | `precursor_concentration`, default 5 |
| BBa_0004 | LacI, GFP | 1 | `inducer_concentration`, default 1 IPTG |
| BBa_0005 | TetR, GFP, cI, LacI | 1 | `inducer_concentration`, default 1 aTc |
| BBa_I5200 | cI, GFP, LacI, TetR | 1 | none |

Parameters are JSON numbers:

```console
uv run cm run \
  --model examples/tutorials/simbol_circuits.py \
  --parameter circuit='"bba_0004"' \
  --parameter inducer_concentration=4.0 \
  --seed 42 \
  --steps 500 \
  --dt 0.01 \
  --output results/bba-0004.cm2.json
```

## Circuit equations

The ports retain the generated scripts’ default production, degradation,
threshold, and Hill-coefficient values. With

```text
repress(r) = 2^4 / (2^4 + r^4),
```

the non-signaling models are:

### BBa_0001: constitutive GFP

```text
dGFP/dt = 1 - 0.05 GFP.
```

### BBa_0002: TetR-repressed RFP

```text
active_TetR = TetR repress(aTc)
dRFP/dt  = repress(active_TetR) - 0.05 RFP
dTetR/dt = 1 - 0.05 TetR.
```

### BBa_0004: LacI-repressed GFP

```text
active_LacI = LacI repress(IPTG)
dLacI/dt = 2 - 0.1 LacI
dGFP/dt  = 2 repress(active_LacI) - 0.1 GFP.
```

This follows the generated script’s explicit choice to make LacI
constitutive, even though the summarized SBOL fixture places LacI and GFP under
the same LacI-repressed promoter. That source discrepancy is preserved as a
documented tutorial assumption rather than hidden.

### BBa_0005: aTc/TetR, cI, LacI, and GFP network

TetR is constitutive; active TetR represses cI. cI and LacI jointly repress
both GFP and LacI production through the product of two repression factors.

### BBa_I5200: repressilator-like network

TetR represses cI, cI represses LacI, and LacI represses TetR and GFP. All
maximal production rates are two and all first-order degradation rates are
0.1. This is the topology encoded by the fixture and generator; it is not a
claim that the selected constants reproduce a calibrated repressilator.

## BBa_0003: extracellular quorum response

BBa_0003 adds one extracellular LuxR–AHL signal. LuxR and LuxI are
constitutively produced. Their intracellular complex pool is produced in
proportion to `(LuxR + LuxI) * precursor_concentration`; the extracellular
signal activates GFP through a fourth-order Hill function.

The membrane amount flux is

```text
J = 0.1 (complex_pool - extracellular_signal) cell_surface_area.
```

The intracellular complex loses `J / cell_volume`; the grid receives `J` as
an amount-per-time source and performs its own voxel-volume conversion. This
uses CellModeller2’s declared conservation convention instead of embedding
`gridVolume` divisions in generated source.

The SimBOL output declares a `400 x 400 x 3` grid at spacing one. The tutorial
port uses `80 x 80 x 3`, still centered on the founder, so ordinary runs remain
bounded. Grid extent affects escape to boundaries and must be restored or
convergence-tested for a large-colony study.

## Reviewed semantic differences

The generated non-signaling scripts update arbitrary Python attributes once
per callback and then halve those attributes at division. They call the values
“concentrations,” but halving a daughter concentration does not conserve amount
when daughter volumes sum to the parent volume. The native ports use the
CellModeller2 concentration contract instead:

- rates are multiplied by the explicit simulation `dt`;
- volume growth dilutes concentration;
- division copies concentration and conserves amount; and
- rate equations are simultaneous rather than Python assignment ordered.

The generated scripts also implement aTc or IPTG by multiplying the repressor
concentration once per callback. That makes inducer action depend on step count
and destroys the modeled repressor. The ports interpret inducer binding as an
algebraic active-repressor fraction used by the downstream Hill repression,
while total repressor follows its production/degradation ODE.

These are scientific-model corrections, not numerical equivalence claims. A
study that requires exact reproduction of generated callback trajectories
should run and record the legacy source separately.

## Danino quorum-sensing clock

```console
uv run cm view \
  --model examples/tutorials/danino_clock.py \
  --seed 42 \
  --dt 0.01 \
  --open
```

The port retains:

- LuxI, AiiA, and GFP intracellular channels;
- shared extracellular AHL;
- AHL-activated production with a third-order Hill response;
- LuxI-dependent AHL production and AiiA-dependent AHL removal;
- stochastic daughter perturbations; and
- the finite trap/channel obstacle geometry, expressed with typed plane and
  outside-sphere constraints.

The biological motif is based on Danino et al., “A synchronized quorum of
genetic clocks,” Nature 463, 326–330 (2010), as cited by the SimBOL model. The
example equations and constants are a tutorial realization, not a reproduction
of the paper’s experimental parameter inference.

### Deliberate transport boundary

`CM_Danino.py` subclasses the legacy grid to apply an x-dependent AHL sink and
an x-dependent nutrient source/decay field. It then reads nutrient to regulate
growth. CellModeller2’s current typed coupled plan supports diffusion,
advection, boundaries, cell sampling, and cell-scattered sources; it does not
yet declare arbitrary field-wide reaction masks.

The native tutorial therefore ports the core AHL oscillator and trap with
constant configured growth. It does not emulate the nutrient mask on the host,
silently drop it while retaining nutrient-dependent growth, or claim numerical
equivalence to the full custom subclass. A complete follow-up should add a
typed, checkpointed, backend-conformant grid-reaction representation before
restoring those terms.

## Exercises

- Run an inducer sweep with a data-only run manifest and compare reporter
  concentration at a fixed physical time and cell count.
- Compare BBa_0004 with a self-repressed LacI equation derived directly from
  the summarized SBOL topology. Treat it as a different model, not a bug-free
  rerun of the generated script.
- Restore the larger BBa_0003 grid and perform a grid-extent convergence check.
- Add explicit parameter names and units to a copied circuit before fitting
  data. The SimBOL defaults are dimensionless tutorial values.
- Design a versioned SimBOL intermediate schema and fail explicitly on SBOL
  interactions that cannot be compiled to the bounded rate-plan operations.

