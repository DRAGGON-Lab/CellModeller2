# Legacy fixed-position audit

## Scope

The audited implementation is `CellModeller/Biophysics/GeneralModels/CLFixedPosition.py` in legacy CellModeller. Repository search found no bundled model that selects this module.

## Observed behavior

`CLFixedPosition` is a complete alternate biophysics implementation rather than a flag on the ordinary rod model. Its device arrays contain center coordinates, current and previous scalar volumes, and volume growth rates. Adding a cell copies `state.pos`, `state.volume`, and `state.growthRate` into those arrays. The step computes

```text
growth_rate = state.growthRate * state.volume
volume_next = volume + dt * growth_rate
```

and never updates a position. It has no direction, capsule length, radius, contact graph, or mechanical relaxation. Its host-state update copies volume and position back to the legacy cell object.

## Compatibility judgment

The reusable semantic requirement is “biological state advances while the cell is mechanically immovable.” The scalar point-volume representation is not isomorphic to CellModeller2's rod length and radius: a conversion would need to choose a shape and specify whether radius, total capsule length, or cylindrical length remains fixed. Silently equating volume with rod length would change both growth and contact geometry.

CellModeller2 therefore represents fixedness as a persistent rod-cell attribute. Its normal length-growth, species, signaling, and division stages continue, while the mechanics system projects that cell's correction degrees of freedom to zero. The ordinary data-only checkpoint records the attribute. No generic legacy volume conversion is claimed.
