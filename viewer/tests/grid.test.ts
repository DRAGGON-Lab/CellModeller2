import { describe, expect, it } from "vitest";

import { flatSignalIndex, signalSlice, sliceDimension } from "../src/grid";
import type { SceneGridBoundary, SceneSignalGrid } from "../src/scene";

const boundary: SceneGridBoundary = { kind: "no_flux", values: [] };
const grid: SceneSignalGrid = {
  signalCount: 2,
  shape: [2, 3, 2],
  origin: [10, 20, 30],
  spacing: [0.5, 1, 2],
  boundaries: {
    xLower: boundary,
    xUpper: boundary,
    yLower: boundary,
    yUpper: boundary,
    zLower: boundary,
    zUpper: boundary,
  },
  levels: Array.from({ length: 24 }, (_, index) => index),
};

describe("signal grid slicing", () => {
  it("uses the engine's channel-major x-y-z indexing", () => {
    expect(flatSignalIndex(grid, 0, 1, 2, 1)).toBe(11);
    expect(flatSignalIndex(grid, 1, 1, 2, 1)).toBe(23);
  });

  it("extracts an XY slice with x varying across texture rows", () => {
    const slice = signalSlice(grid, 1, "z", 1);
    expect(slice.width).toBe(2);
    expect(slice.height).toBe(3);
    expect(slice.values).toEqual([13, 19, 15, 21, 17, 23]);
    expect(slice.center).toEqual([10.25, 21, 32]);
    expect(slice.horizontalSpan).toBe(0.5);
    expect(slice.verticalSpan).toBe(2);
  });

  it("reports dimensions and rejects invalid coordinates", () => {
    expect(sliceDimension(grid, "x")).toBe(2);
    expect(sliceDimension(grid, "y")).toBe(3);
    expect(sliceDimension(grid, "z")).toBe(2);
    expect(() => signalSlice(grid, 0, "x", 2)).toThrow("out of range");
    expect(() => flatSignalIndex(grid, 2, 0, 0, 0)).toThrow("out of range");
  });
});
