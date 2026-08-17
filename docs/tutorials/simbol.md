# SimBOL circuit examples

SimBOL connects an SBOL 3 design to simulator-specific code through a summarized JSON representation. This tutorial presents typed CellModeller2 versions of six BioBrick circuit examples and a spatial quorum-sensing clock.

These are explicit example models, not a general SBOL-to-rate-plan import path. The [source reference](../compatibility/tutorial-source-provenance.md#simbol-source-workflow) describes how they relate to the SimBOL notebook, generated Python, and JSON fixtures.

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

Allowed names are `bba_0001`, `bba_0002`, `bba_0003`, `bba_0004`, `bba_0005`, and `bba_i5200`. Choose `Species` coloring and the reporter channel listed below.

| Circuit | Ordered species channels | Reporter channel | Additional input |
| --- | --- | --: | --- |
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

The examples use the generated scripts’ default production, degradation, threshold, and Hill-coefficient values. With

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

For BBa_0004, LacI is constitutive because that is the equation in the generated script. The summarized SBOL fixture instead places LacI and GFP under the same LacI-repressed promoter, so readers deriving equations directly from the SBOL topology should treat that as a different model.

### BBa_0005: aTc/TetR, cI, LacI, and GFP network

TetR is constitutive; active TetR represses cI. cI and LacI jointly repress both GFP and LacI production through the product of two repression factors.

### BBa_I5200: repressilator-like network

TetR represses cI, cI represses LacI, and LacI represses TetR and GFP. All maximal production rates are two and all first-order degradation rates are 0.1. This is the topology encoded by the fixture and generator; it is not a claim that the selected constants reproduce a calibrated repressilator.

## BBa_0003: extracellular quorum response

BBa_0003 adds one extracellular LuxR–AHL signal. LuxR and LuxI are constitutively produced. Their intracellular complex pool is produced in proportion to `(LuxR + LuxI) * precursor_concentration`; the extracellular signal activates GFP through a fourth-order Hill function.

The membrane amount flux is

```text
J = 0.1 (complex_pool - extracellular_signal) cell_surface_area.
```

The intracellular complex loses `J / cell_volume`; the grid receives `J` as an amount-per-time source and performs its own voxel-volume conversion. This uses CellModeller2’s declared conservation convention instead of embedding `gridVolume` divisions in generated source.

The SimBOL output declares a `400 x 400 x 3` grid at spacing one. The tutorial uses `80 x 80 x 3`, still centered on the founder, so ordinary runs remain bounded. Grid extent affects escape to boundaries and should be restored or convergence-tested for a large-colony study.

## Modeling conventions and source differences

The generated non-signaling scripts update arbitrary Python attributes once per callback and then halve those attributes at division. They call the values “concentrations,” but halving a daughter concentration does not conserve amount when daughter volumes sum to the parent volume. The native ports use the CellModeller2 concentration contract instead:

- rates are multiplied by the explicit simulation `dt`;
- volume growth dilutes concentration;
- division copies concentration and conserves amount; and
- rate equations are simultaneous rather than Python assignment ordered.

The generated scripts also implement aTc or IPTG by multiplying the repressor concentration once per callback. That makes inducer action depend on step count and destroys the modeled repressor. These examples instead treat inducer binding as an algebraic active-repressor fraction used by the downstream Hill repression, while total repressor follows its production/degradation ODE.

These choices change trajectories relative to the generated callback scripts. A study that requires exact reproduction of those scripts should run and record the original source separately.

## Danino quorum-sensing clock

```console
uv run cm view \
  --model examples/tutorials/danino_clock.py \
  --seed 42 \
  --dt 0.01 \
  --open
```

The example includes:

- LuxI, AiiA, and GFP intracellular channels;
- shared extracellular AHL and nutrient fields;
- AHL-activated production with a third-order Hill response;
- LuxI-dependent AHL production and AiiA-dependent AHL removal;
- a flow-fed channel that delivers nutrient, carries secreted AHL downstream, and washes out escaped cells;
- nutrient-limited growth from the sampled local field;
- stochastic daughter perturbations; and
- the device geometry, flow field, obstacle mask, and inlet/outlet built from one `TrapChannelDevice` description in `cellmodeller2.microfluidics`.

The biological motif is based on Danino et al., “A synchronized quorum of genetic clocks,” Nature 463, 326–330 (2010), as cited by the SimBOL model. The example equations and constants are a tutorial realization, not a reproduction of the paper’s experimental parameter inference.

### Device flow and washout

`CM_Danino.py` subclasses the legacy grid to fake the channel with an x-dependent AHL sink and
nutrient source field. The CellModeller2 model expresses the channel physically: a
`TrapChannelDevice` projects one geometry description into box wall constraints, a signal-grid
obstacle mask, a divergence-free Poiseuille flow profile along the channel, and fixed inlet
and outlet boundaries. Nutrient enters at the inlet at concentration 10 and is carried past
the trap mouth; AHL secreted by the colony diffuses out of the trap and is advected
downstream; walls block both diffusion and advection.

Cells feel the same flow: the controller enables `flow_drift`, so a cell that escapes the
trap is carried along the channel, and the regulation step removes any cell past the washout
boundary with a `StepPlan` removal, forgetting its division target first.

Before each biological step, the controller samples nutrient channel 1 at each cell center
and applies the saturating growth law:

```text
growth = nutrient / (5 + nutrient).
```

The device starts flooded with media, matching how a physical device is loaded before flow
begins.

### Numerical interpretation

Forward Euler evaluates transport, affine field reaction, and cell scatter from the old field and commits them together. Its preflight stability bound includes the largest local loss rate for each signal. This model selects Crank–Nicolson: spatial losses enter the implicit diagonal, fixed sources enter both trapezoidal halves, and cell-scattered AHL exchange remains an old-field explicit source. A converged negative result is still rejected because Crank–Nicolson is not positivity preserving for arbitrarily stiff steps.

## Exercises

- Vary `mean_flow_speed` and measure how the trap's AHL retention, and therefore the clock's synchronization, responds.
- Run an inducer sweep with a data-only run manifest and compare reporter concentration at a fixed physical time and cell count.
- Compare BBa_0004 with a self-repressed LacI equation derived directly from the summarized SBOL topology. Treat it as a different model, not a bug-free rerun of the generated script.
- Restore the larger BBa_0003 grid and perform a grid-extent convergence check.
- Add explicit parameter names and units to a copied circuit before fitting data. The SimBOL defaults are dimensionless tutorial values.
