# Legacy analysis and utility script audit

This audit covers the legacy `Scripts/` directory and the code cells in `Scripts/Analysis.ipynb`. The scripts are evidence about past workflows, not a single compatibility API. Several are Python 2-era, execute trusted pickle content, reach into OpenCL solver buffers, or encode workstation-specific paths.

## Inventory

| Source | Observed behavior | CellModeller2 disposition |
| --- | --- | --- |
| `batch.py` | selects an OpenCL platform/device interactively, then runs until a fixed cell-capacity margin | replace device selection with `cm devices`; add a declared cell-count stopping criterion rather than capacity coupling |
| `multi_batch.py` | repeats the same unseeded run three times | replace with explicit run manifests whose replicate IDs, seeds, parameters, and outputs are reviewable data |
| `batch_iter.py` | hard-coded gamma sweep over eight values | replace with manifest generation plus ordinary `cm run` jobs; a scheduler is not part of the simulator |
| `batchFile.py` | cluster-specific model defaults, timestamped pickle directories, and private physics setup calls | retire; checkpoints, provenance, and explicit output paths supersede it |
| `CellModellerGUI.py` | starts the PyQt/OpenGL GUI | replaced by the independent scene viewer and live controller |
| `LengthHistogram.py` | writes radial position and full capsule length to a global CSV | preserve the two quantities in typed cell tables; the old `length + 2 * radius` value must be named `capsule_length` |
| `spatial_analyze.py` | computes counts and mean species channel 0 in 20 radial XY bins | provide a documented dataframe recipe; do not preserve its inconsistent bin coordinates or hard-coded radius and channel |
| `contactGraph.py` | reloads executable model text, reconstructs OpenCL contacts, collapses geometric rows into an undirected graph, and draws a PDF | export typed contact rows directly from `Simulation.find_cell_contacts`; derive unique neighbor edges as a query, without re-executing model source |
| `Draw2DPDF.py` | renders 2D capsules and one signal slice from pickle fields | scene v1 preserves the required geometry and signals; publication rendering is a viewer/export concern, not an analysis-table contract |
| `video.sh`, `contactVideo.sh` | shell loops through pickles, PDF conversion, and ffmpeg | replace later with an explicit scene-frame rendering/export workflow; do not make ImageMagick filename conventions part of the engine |
| `printTiming.py` | prints cell count beside checkpoint filesystem modification time | retire; file mtime is neither simulation time nor measured runtime |
| `gitPublish.sh` | merges a historical private repository into a public repository | retire; unrelated to simulation or analysis semantics |
| `Analysis.ipynb` | demonstrates position/orientation plots, length and radial histograms, length-weighted 2D density, raw overlap inspection, lineage, and sister-neighbor counting | publish small recipes over versioned datasets; private solver buffers and hard-coded local paths are not supported analysis APIs |

## Scientific quantities worth preserving

The legacy scripts imply a compact, useful analysis vocabulary:

- simulation time, source checkpoint, backend identity, and cell count per frame;
- stable cell ID, active parent ID, compact slot, position, direction, cylinder length, radius, full capsule length, growth rate, cell type, fixed state, and every species channel;
- every geometric contact row, including its two stable IDs, ordinal, surface point, normal, signed separation, and weight;
- unique neighbor edges derived by grouping contact rows by stable ID pair;
- signal-grid values with explicit channel and x-y-z coordinates; and
- experiment identity, seed, parameters, model digest, and checkpoint digest.

The old notebook's density example weights XY histogram bins by full capsule length. That is a declared line-density proxy, not cell area, volume, biomass, or packing fraction. Any replacement recipe must use that name. Likewise, negative signed separation is penetration; an `overlap` column may be derived as `max(0, -signed_separation)` but must not replace the signed value.

The radial species script contains no trustworthy bin contract. Its plotted coordinates start at zero, while its text output uses half of each upper edge rather than the actual bin center. CellModeller2 will define bins through explicit edges and report their left edge, right edge, and center.

## Unsafe or non-portable behavior not preserved

- Python pickle remains confined to the explicit trusted one-way importer.
- Analysis never imports model source from a checkpoint or invokes callbacks.
- Private OpenCL arrays and solver methods are not an analysis API.
- Stable IDs are never replaced by compact slots in exported relationships.
- File modification time is not accepted as a physical or performance clock.
- Hard-coded paths, implicit working-directory outputs, and unsorted glob loops are not reproducibility contracts.
- Legacy per-cell RGB callback attributes are not scientific state. Color maps remain declarative presentation choices.

## Acceptance boundary

Analysis compatibility is satisfied when CellModeller2 can convert an explicit, ordered checkpoint series into versioned Parquet cell/contact tables and Zarr signal arrays, and the documented dataframe recipes reproduce the meaningful legacy quantities above. CPU, Metal, and CUDA checkpoints use the same storage schema. Derived contact rows record the backend and contact parameters used; the CUDA development workflow exercises both the native contact fixture and a derived-contact export.

Offline publication rendering and video export are separately useful, but are not prerequisites for scientific analysis-table compatibility.
