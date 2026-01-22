import { describe, it, expect } from "vitest";
import { applyJsonPatches, ensurePatches } from "../src/trustcall/json-patch.js";
import type { JsonPatchOp } from "../src/trustcall/types.js";

describe("applyJsonPatches", () => {
  describe("add operation", () => {
    it("should add a new property to an object", () => {
      const target = { name: "Alice" };
      const patches: JsonPatchOp[] = [{ op: "add", path: "/age", value: 30 }];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ name: "Alice", age: 30 });
    });

    it("should add a nested property", () => {
      const target = { user: { name: "Alice" } };
      const patches: JsonPatchOp[] = [
        { op: "add", path: "/user/age", value: 30 },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ user: { name: "Alice", age: 30 } });
    });

    it("should create intermediate objects when needed", () => {
      const target = {};
      const patches: JsonPatchOp[] = [
        { op: "add", path: "/user/name", value: "Alice" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ user: { name: "Alice" } });
    });

    it("should append to array with - index", () => {
      const target = { items: [1, 2, 3] };
      const patches: JsonPatchOp[] = [
        { op: "add", path: "/items/-", value: 4 },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ items: [1, 2, 3, 4] });
    });

    it("should add to specific array index", () => {
      const target = { items: ["a", "b", "c"] };
      const patches: JsonPatchOp[] = [
        { op: "add", path: "/items/1", value: "x" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ items: ["a", "x", "c"] });
    });
  });

  describe("remove operation", () => {
    it("should remove a property from an object", () => {
      const target = { name: "Alice", age: 30 };
      const patches: JsonPatchOp[] = [{ op: "remove", path: "/age" }];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ name: "Alice" });
    });

    it("should remove a nested property", () => {
      const target = { user: { name: "Alice", age: 30 } };
      const patches: JsonPatchOp[] = [{ op: "remove", path: "/user/age" }];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ user: { name: "Alice" } });
    });

    it("should remove an array element", () => {
      const target = { items: ["a", "b", "c"] };
      const patches: JsonPatchOp[] = [{ op: "remove", path: "/items/1" }];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ items: ["a", "c"] });
    });
  });

  describe("replace operation", () => {
    it("should replace an existing property", () => {
      const target = { name: "Alice", age: 30 };
      const patches: JsonPatchOp[] = [
        { op: "replace", path: "/age", value: 31 },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ name: "Alice", age: 31 });
    });

    it("should replace a nested property", () => {
      const target = { user: { name: "Alice" } };
      const patches: JsonPatchOp[] = [
        { op: "replace", path: "/user/name", value: "Bob" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ user: { name: "Bob" } });
    });

    it("should replace an array element", () => {
      const target = { items: ["a", "b", "c"] };
      const patches: JsonPatchOp[] = [
        { op: "replace", path: "/items/1", value: "x" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ items: ["a", "x", "c"] });
    });
  });

  describe("move operation", () => {
    it("should move a property", () => {
      const target = { oldName: "value", other: "data" };
      const patches: JsonPatchOp[] = [
        { op: "move", path: "/newName", from: "/oldName" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ newName: "value", other: "data" });
    });

    it("should move an array element", () => {
      const target = { items: ["a", "b", "c"] };
      const patches: JsonPatchOp[] = [
        { op: "move", path: "/first", from: "/items/0" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ items: ["b", "c"], first: "a" });
    });
  });

  describe("copy operation", () => {
    it("should copy a property", () => {
      const target = { original: "value" };
      const patches: JsonPatchOp[] = [
        { op: "copy", path: "/duplicate", from: "/original" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ original: "value", duplicate: "value" });
    });

    it("should deep copy objects", () => {
      const target = { original: { nested: "value" } };
      const patches: JsonPatchOp[] = [
        { op: "copy", path: "/duplicate", from: "/original" },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({
        original: { nested: "value" },
        duplicate: { nested: "value" },
      });
      // Verify it's a deep copy
      expect(result.original).not.toBe(result.duplicate);
    });
  });

  describe("test operation", () => {
    it("should pass when value matches", () => {
      const target = { name: "Alice" };
      const patches: JsonPatchOp[] = [
        { op: "test", path: "/name", value: "Alice" },
      ];
      expect(() => applyJsonPatches(target, patches)).not.toThrow();
    });

    it("should throw when value does not match", () => {
      const target = { name: "Alice" };
      const patches: JsonPatchOp[] = [
        { op: "test", path: "/name", value: "Bob" },
      ];
      expect(() => applyJsonPatches(target, patches)).toThrow(
        'Test failed: expected "Bob", got "Alice"'
      );
    });
  });

  describe("multiple patches", () => {
    it("should apply multiple patches in order", () => {
      const target = { a: 1 };
      const patches: JsonPatchOp[] = [
        { op: "add", path: "/b", value: 2 },
        { op: "add", path: "/c", value: 3 },
        { op: "replace", path: "/a", value: 10 },
      ];
      const result = applyJsonPatches(target, patches);
      expect(result).toEqual({ a: 10, b: 2, c: 3 });
    });
  });

  describe("immutability", () => {
    it("should not modify the original object", () => {
      const target = { name: "Alice", nested: { value: 1 } };
      const patches: JsonPatchOp[] = [
        { op: "replace", path: "/name", value: "Bob" },
        { op: "replace", path: "/nested/value", value: 2 },
      ];
      const result = applyJsonPatches(target, patches);
      expect(target).toEqual({ name: "Alice", nested: { value: 1 } });
      expect(result).toEqual({ name: "Bob", nested: { value: 2 } });
    });
  });
});

describe("ensurePatches", () => {
  it("should return patches array directly", () => {
    const patches: JsonPatchOp[] = [{ op: "add", path: "/foo", value: "bar" }];
    const result = ensurePatches({ patches });
    expect(result).toEqual(patches);
  });

  it("should parse patches from JSON string", () => {
    const patches: JsonPatchOp[] = [{ op: "add", path: "/foo", value: "bar" }];
    const result = ensurePatches({ patches: JSON.stringify(patches) });
    expect(result).toEqual(patches);
  });

  it("should extract patches from string with embedded array", () => {
    const patches: JsonPatchOp[] = [{ op: "add", path: "/foo", value: "bar" }];
    const result = ensurePatches({
      patches: `Here are the patches: ${JSON.stringify(patches)} done`,
    });
    expect(result).toEqual(patches);
  });

  it("should return empty array for invalid input", () => {
    expect(ensurePatches({ patches: "not valid" })).toEqual([]);
    expect(ensurePatches({ patches: null })).toEqual([]);
    expect(ensurePatches({ patches: undefined })).toEqual([]);
    expect(ensurePatches({})).toEqual([]);
  });

  it("should return empty array for non-array JSON", () => {
    const result = ensurePatches({ patches: '{"not": "array"}' });
    expect(result).toEqual([]);
  });
});
