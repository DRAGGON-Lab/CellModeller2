# Microfluidic devices: walls, flow, and washout

This tutorial builds models that live inside devices: geometry that confines cells, blocks
chemistry, and carries media. Three examples cover the range:

| Model | Device | Demonstrates |
| --- | --- | --- |
| [`examples/culture_dish.py`](../../examples/culture_dish.py) | round dish | one inside-cylinder constraint as a dish |
| [`examples/microfluidic_trap.py`](../../examples/microfluidic_trap.py) | trap + channel | flow, obstacles, drift, washout |
| [`examples/tutorials/danino_clock.py`](../../examples/tutorials/danino_clock.py) | trap + channel | the full quorum clock in a device |
| [`examples/tutorials/biopixel_trap.py`](../../examples/tutorials/biopixel_trap.py) | biopixel array trap | fabricated dimensions, monolayer cavity |

Run any of them live:

```console
uv run cm view --model examples/microfluidic_trap.py --seed 42 --dt 0.02 --backend metal --open
```

## Walls that cells and chemistry both respect

Mechanical walls are typed external constraints: infinite planes, spheres, axis-aligned
boxes, and z-aligned cylinders, each with an inside or outside permitted region. A round
culture dish is a single inside cylinder whose barrel is the wall and whose caps confine the
monolayer:

```python
dish = CylinderConstraintInit()
dish.radius = 30.0
dish.half_height = 1.0
dish.allowed_region = ConstraintRegion.INSIDE
simulation.add_cylinder_constraint(dish)
```

Constraints alone are invisible to signals. The signal grid's obstacle mask closes every
lattice face between fluid and solid voxels, so diffusion and advection stop at walls, and
sampling near a wall renormalizes over fluid sites. Keeping the mask consistent with the
constraints is an authoring concern, which the device helpers handle.

## Devices from one description

`cellmodeller2.microfluidics.TrapChannelDevice` describes an open-sided trap fed by a
straight channel and projects that one description into every engine input:

```python
from cellmodeller2.microfluidics import TrapChannelDevice

DEVICE = TrapChannelDevice(mean_flow_speed=20.0)
DEVICE.add_constraints(simulation)                     # box walls for mechanics
DEVICE.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0])
```

`apply_to_grid` materializes the solid mask, fixed inlet and outlet boundaries on the y
axis, and the numerically solved steady device flow on the grid's face-staggered velocity
field (see the next section). Flow runs through the channel, circulates weakly at the open
trap face, and the dead-end trap exchanges with the channel chiefly by diffusion, as in the
physical device.

## Flow on signals and on cells

The velocity field advects every signal with conservative upwind face fluxes under both
integrators. Cells feel the same field through explicit drift: with
`MechanicsConfig(flow_drift=True)`, the controller advects each non-fixed cell by the fluid
velocity sampled at its endpoints before contact relaxation, so escaped cells travel down the
channel and rods rotate in shear. Contact relaxation then resolves any overlap the drift
produced against walls or neighbors.

## Washout

Cells that reach the end of the channel leave the system through plan removals:

```python
def _regulate(step):
    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if abs(cell.position.y) > WASHOUT_Y)
    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(r for r in divisions if r.parent_id not in washed)
    return StepPlan(updates=..., divisions=divisions, removals=washed)
```

Removal keeps stable identifiers and lineage history, so analysis can count washout events
and trace removed cells' ancestry from checkpoints.

## Numerical flow: arbitrary geometry and colony feedback

Device flow fields are solved, not authored: `cellmodeller2.flow` computes the steady
Hele-Shaw–Brinkman problem over the grid's fluid voxels and returns the same face-staggered
field the engine consumes. `apply_to_grid` runs this solve for every device, and it works
for any mask geometry — junctions, bends, pillars, a CAD-derived layout — not just straight
channels. The solver is also available directly for grids built without a device helper:

```python
from cellmodeller2.flow import colony_mobility, solve_flow_field

field, report = solve_flow_field(grid, mean_inlet_speed=20.0)   # Stokes limit
grid.velocity_field = field
```

The solve is a variable-coefficient pressure problem (`div(m grad p) = 0`), so the returned
fluxes conserve mass per voxel and vanish on wall faces by construction; the flow-axis
boundaries must be `FIXED` to act as inlet and outlet, and the linear solution is rescaled to
the requested mean inlet speed. With uniform mobility this is the Stokes limit of the
depth-averaged closure — correct routing through any mask, plug profile across the channel
width (side-wall boundary layers, of order the gap height, are outside the closure).

The mobility field is where Brinkman feedback enters: `colony_mobility` rasterizes the
colony's volume fraction and adds Kozeny–Carman style drag, so media diverts around a packed
trap and seeps through its edges. Because the field is data, regulation code re-solves as
the colony grows and swaps it into the running simulation — the trap models do this every
`RESOLVE_INTERVAL` steps:

```python
def _regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
        mobility = colony_mobility(GRID, step.cells, drag_coefficient=DRAG_COEFFICIENT)
        field, _ = solve_flow_field(GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility)
        step.simulation.set_velocity_field(field)
    ...
```

`Simulation.set_velocity_field` validates the replacement against the full grid
specification before swapping it; transport, drift, and checkpoints all use whichever field
is current. The re-solve cadence is a model choice — colony growth is slow, so hundreds of
steps between solves is typical. The drag coefficient is a modeling parameter (how strongly
a packed colony resists through-flow relative to the open channel), not a measured constant.

## A fabricated device: the Prindle biopixel array

The photomask CAD for a real sensing-array device is in
[`docs/tutorials/devices`](devices) as `prindle.dwg` and `prindle.dxf`
(AutoCAD 2018; the DXF opens in most CAD viewers). The array tiles hundreds of
identical traps along media channels. `BiopixelTrapDevice` models one trap with
the fabricated dimensions:

- a cavity 100 x 95 micrometers, only **1.65 micrometers tall**, squeezing the
  colony into a monolayer — the imaging geometry the device is fabricated for;
- the flow layer's 10-micrometer-tall channel along the cavity's open face,
  carrying the media stream past the trap mouth;
- trap dimensions read from the mask at build time:
  `BiopixelTrapDevice.from_mask` takes the drawn 110 x 100 outer wall outline
  and removes the 5-micrometer walls (two sides, one back; the fourth side is
  the open face), recovering the 100 x 95 cavity.

```python
from cellmodeller2.microfluidics import BiopixelTrapDevice

DEVICE = BiopixelTrapDevice(mean_flow_speed=20.0)
```

### Reading the array from the mask

`cellmodeller2.masks` extracts the trap layout directly from the DXF: it parses
the model-space `LWPOLYLINE` outlines (a bounded, data-only reader; block
definitions, which hold orphaned array remnants in mask files, are ignored) and
selects axis-aligned rectangles by layer and size. Mask drawings use one unit
per millimeter, so `unit_scale=1000.0` yields micrometers:

```python
from cellmodeller2.masks import extract_rectangles, load_mask_polylines, match_rectangles

polylines = load_mask_polylines("docs/tutorials/devices/prindle.dxf")
rectangles = extract_rectangles(polylines, layer="Layer-2", unit_scale=1000.0)
traps = match_rectangles(rectangles, 110.0, 100.0, tolerance=1.0)
```

On this mask that yields the full biopixel array: **496 traps in a 16 x 31
grid** at a 172.5 x 125 micrometer pitch, spanning 2.4 x 3.75 millimeters. The
drawn outline is the 110 x 100 micrometer outer wall of the 100 x 95 cavity.
With ``include_blocks=True`` the reader also traverses block definitions,
where this mask keeps its flow layer: `Layer-5` carries supply channels about
one millimeter wide plus 60-micrometer post features, on a separately laid-out
plate (each mask layer is its own plate, registered only lithographically, so
cross-layer alignment is not recoverable from the drawing).
Each trap's `center` places a `BiopixelTrapDevice` on the physical die, so an
array study can iterate the extracted centers while simulating one trap at a
time.

Because every trap in the array sees the same inlet media and flow, one
simulated trap is representative of each biopixel; running the single-trap
model is running the array, one pixel at a time. Inter-trap coupling (the
array's gas-phase synchronization) is not part of this model. Run it live:

```console
uv run cm view --model examples/tutorials/biopixel_trap.py --seed 5 --dt 0.02 --backend metal --open
```

The viewer shows the shallow cavity as a thin glass slab beside the taller
channel of the flow layer, with the monolayer spreading at mid-cavity height
and fed through the open trap mouth.

## Numerical guidance

- Choose `dt` so the largest per-step drift, `max_velocity * dt`, stays below a cell radius;
  `solve_flow_field` reports `max_speed`, and the trap examples use `dt = 0.02` with a mean
  channel speed of 20.
- Forward Euler enforces its stability bound from the per-site advective outflow; the trap
  models select Crank-Nicolson.
- Prime the device with media in its initial levels when growth should start immediately; an
  unprimed trap fills by diffusion on the timescale `L^2 / D`.
- A sampling position inside a wall raises an error rather than returning zero; regulation
  code samples at cell centers, which mechanics keeps out of walls.
