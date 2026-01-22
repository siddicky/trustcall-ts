import { describe, it, expect } from "vitest";
import {
  JsonPatchSchema,
  PatchFunctionErrorsSchema,
  PatchDocSchema,
  createRemoveDocSchema,
  createPatchFunctionNameSchema,
} from "../src/trustcall/schemas.js";

describe("JsonPatchSchema", () => {
  it("should accept valid add operation", () => {
    const patch = { op: "add", path: "/foo", value: "bar" };
    const result = JsonPatchSchema.parse(patch);
    expect(result).toEqual(patch);
  });

  it("should accept valid remove operation", () => {
    const patch = { op: "remove", path: "/foo" };
    const result = JsonPatchSchema.parse(patch);
    expect(result).toEqual(patch);
  });

  it("should accept valid replace operation", () => {
    const patch = { op: "replace", path: "/foo", value: "bar" };
    const result = JsonPatchSchema.parse(patch);
    expect(result).toEqual(patch);
  });

  it("should reject invalid operation", () => {
    const patch = { op: "invalid", path: "/foo" };
    expect(() => JsonPatchSchema.parse(patch)).toThrow();
  });

  it("should require path", () => {
    const patch = { op: "add", value: "bar" };
    expect(() => JsonPatchSchema.parse(patch)).toThrow();
  });
});

describe("PatchFunctionErrorsSchema", () => {
  it("should accept valid patch function errors", () => {
    const data = {
      json_doc_id: "test-id",
      planned_edits: "- Fix field X\n- Add field Y",
      patches: [{ op: "add", path: "/foo", value: "bar" }],
    };
    const result = PatchFunctionErrorsSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should require json_doc_id", () => {
    const data = {
      planned_edits: "some edits",
      patches: [],
    };
    expect(() => PatchFunctionErrorsSchema.parse(data)).toThrow();
  });

  it("should require planned_edits", () => {
    const data = {
      json_doc_id: "test-id",
      patches: [],
    };
    expect(() => PatchFunctionErrorsSchema.parse(data)).toThrow();
  });

  it("should require patches array", () => {
    const data = {
      json_doc_id: "test-id",
      planned_edits: "some edits",
    };
    expect(() => PatchFunctionErrorsSchema.parse(data)).toThrow();
  });

  it("should validate patches within array", () => {
    const data = {
      json_doc_id: "test-id",
      planned_edits: "some edits",
      patches: [{ op: "invalid", path: "/foo" }],
    };
    expect(() => PatchFunctionErrorsSchema.parse(data)).toThrow();
  });
});

describe("PatchDocSchema", () => {
  it("should accept valid patch doc", () => {
    const data = {
      json_doc_id: "doc-1",
      planned_edits: "Update name field",
      patches: [{ op: "replace", path: "/name", value: "New Name" }],
    };
    const result = PatchDocSchema.parse(data);
    expect(result).toEqual(data);
  });

  it("should accept empty patches array", () => {
    const data = {
      json_doc_id: "doc-1",
      planned_edits: "No changes needed",
      patches: [],
    };
    const result = PatchDocSchema.parse(data);
    expect(result).toEqual(data);
  });
});

describe("createRemoveDocSchema", () => {
  it("should create schema accepting allowed IDs", () => {
    const schema = createRemoveDocSchema(["doc-1", "doc-2", "doc-3"]);
    const result = schema.parse({ json_doc_id: "doc-1" });
    expect(result).toEqual({ json_doc_id: "doc-1" });
  });

  it("should reject IDs not in allowed list", () => {
    const schema = createRemoveDocSchema(["doc-1", "doc-2"]);
    expect(() => schema.parse({ json_doc_id: "doc-999" })).toThrow(
      "Document ID must be one of: doc-1, doc-2"
    );
  });

  it("should work with single allowed ID", () => {
    const schema = createRemoveDocSchema(["only-one"]);
    const result = schema.parse({ json_doc_id: "only-one" });
    expect(result).toEqual({ json_doc_id: "only-one" });
  });
});

describe("createPatchFunctionNameSchema", () => {
  it("should create schema with valid tool names in description", () => {
    const schema = createPatchFunctionNameSchema(["tool1", "tool2"]);
    const result = schema.parse({
      json_doc_id: "test-id",
      reasoning: ["Reason 1", "Reason 2"],
      fixed_name: "tool1",
    });
    expect(result.fixed_name).toBe("tool1");
  });

  it("should accept schema without fixed_name", () => {
    const schema = createPatchFunctionNameSchema(["tool1"]);
    const result = schema.parse({
      json_doc_id: "test-id",
      reasoning: ["Reason 1", "Reason 2"],
    });
    expect(result.fixed_name).toBeUndefined();
  });

  it("should require reasoning array", () => {
    const schema = createPatchFunctionNameSchema();
    expect(() =>
      schema.parse({
        json_doc_id: "test-id",
        fixed_name: "some-tool",
      })
    ).toThrow();
  });

  it("should work without tool names", () => {
    const schema = createPatchFunctionNameSchema();
    const result = schema.parse({
      json_doc_id: "test-id",
      reasoning: ["Reason 1", "Reason 2"],
      fixed_name: "any-name",
    });
    expect(result.fixed_name).toBe("any-name");
  });
});
