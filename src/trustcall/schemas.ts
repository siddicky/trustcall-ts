import { z } from "zod";

/**
 * JSON Patch schema for validation error fixes.
 */
export const JsonPatchSchema = z.object({
  op: z
    .enum(["add", "remove", "replace"])
    .describe(
      "The operation to be performed. Must be one of 'add', 'remove', 'replace'."
    ),
  path: z
    .string()
    .describe(
      "A JSON Pointer path that references a location within the target document."
    ),
  value: z
    .any()
    .optional()
    .describe(
      "The value to be used within the operation. REQUIRED for 'add', 'replace' operations."
    ),
});

/**
 * Schema for fixing validation errors via patches.
 */
export const PatchFunctionErrorsSchema = z.object({
  json_doc_id: z
    .string()
    .describe("The json_doc_id of the function you are patching."),
  planned_edits: z
    .string()
    .describe(
      "A bullet-point list of each ValidationError and the corresponding JSONPatch operation needed to fix it."
    ),
  patches: z
    .array(JsonPatchSchema)
    .describe(
      "A list of JSONPatch operations to be applied to the previous tool call's arguments."
    ),
});

/**
 * Schema for updating existing documents via patches.
 */
export const PatchDocSchema = z.object({
  json_doc_id: z
    .string()
    .describe("The json_doc_id of the document you are patching."),
  planned_edits: z
    .string()
    .describe(
      "Think step-by-step, reasoning over each required update and the corresponding JSONPatch operation."
    ),
  patches: z
    .array(JsonPatchSchema)
    .describe(
      "A list of JSONPatch operations to be applied to the existing document."
    ),
});

/**
 * Create a schema for removing documents by ID.
 */
export function createRemoveDocSchema(allowedIds: string[]) {
  return z.object({
    json_doc_id: z
      .string()
      .refine((val) => allowedIds.includes(val), {
        message: `Document ID must be one of: ${allowedIds.join(", ")}`,
      })
      .describe(
        `ID of the document to remove. Must be one of: ${allowedIds.join(", ")}`
      ),
  });
}

/**
 * Create a schema for fixing function names.
 */
export function createPatchFunctionNameSchema(validToolNames?: string[]) {
  const nameDescription = validToolNames
    ? ` Must be one of: ${validToolNames.join(", ")}`
    : "";

  return z.object({
    json_doc_id: z
      .string()
      .describe("The json_doc_id of the function you are patching."),
    reasoning: z
      .array(z.string())
      .describe("At least 2 logical reasons why this action should be taken."),
    fixed_name: z
      .string()
      .optional()
      .describe(`The corrected name of the function.${nameDescription}`),
  });
}

export type JsonPatch = z.infer<typeof JsonPatchSchema>;
export type PatchFunctionErrors = z.infer<typeof PatchFunctionErrorsSchema>;
export type PatchDoc = z.infer<typeof PatchDocSchema>;
