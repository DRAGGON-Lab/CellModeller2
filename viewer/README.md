# CellModeller2 scene viewer

This package is a read-only TypeScript and Three.js consumer of
`cellmodeller2-scene` files. It has no Python bridge, model loader, simulation
clock, checkpoint writer, CUDA context, or Metal device access.

## Run locally

From the repository root:

```console
uv run python examples/viewer_scene.py --output viewer-demo.cm2.scene.json
pnpm --dir viewer install
pnpm --dir viewer dev
```

Open the local URL printed by Vite and load `viewer-demo.cm2.scene.json`. The
viewer also accepts a scene by drag and drop.

## Capabilities

- SHA-256 verification over the Python writer's RFC 8785 canonical frame;
- strict scene v1 structural and numerical validation;
- instanced cylinder and sphere rendering for exact spherocylinder geometry;
- orbit, pan, zoom, colony framing, raycast picking, and selection highlighting;
- exact cell geometry, lineage, type, fixed state, growth, and species inspection;
- categorical cell-type and fixed-state color maps;
- perceptual growth-rate and species color maps; and
- selectable signal channel, axis, and grid slice.

Cell IDs remain decimal strings throughout the browser because their unsigned
64-bit range exceeds JavaScript's exact integer range.

## Validate

```console
pnpm --dir viewer format:check
pnpm --dir viewer check
pnpm --dir viewer test
pnpm --dir viewer build
```

The unit suite includes a Python-authored scene fixture whose digest contains
floating-point values that ordinary Python and JavaScript JSON serializers
spell differently. Passing that test is the cross-language integrity gate.
