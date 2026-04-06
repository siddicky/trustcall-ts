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
      "A JSON Pointer path that references a location within the target document where the operation is performed. Note: patches are applied sequentially. If you remove a value, the collection size changes before the next patch is applied."
    ),
  value: z
    .unknown()
    .describe(
      "The value to be used within the operation. REQUIRED for" +
      " 'add', 'replace', and 'test' operations." +
      " Pay close attention to the json schema to ensure" +
      " the patched document will be valid."
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
      "Second, write a bullet-point list of each ValidationError you encountered and the corresponding JSONPatch operation needed to heal it. For each operation, write why your initial guess was incorrect, citing the corresponding types(s) from the JSONSchema that will be used to validate the resultant patched document. Think step-by-step to ensure no error is overlooked."
    ),
  patches: z
    .array(JsonPatchSchema)
    .describe(
      "Finally, provide a list of JSONPatch operations to be applied to the previous tool call's response arguments. If none are required, return an empty list. This field is REQUIRED. Multiple patches in the list are applied sequentially in the order provided, with each patch building upon the result of the previous one."
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
      "Seconds, think step-by-step, reasoning over each required update and the corresponding JSONPatch operation to accomplish it. Cite the fields in the JSONSchema you referenced in developing this plan. Address each path as a group; don't switch between paths. Plan your patches in the following order:1. replace - this keeps collection size the same.2. remove - BE CAREFUL ABOUT ORDER OF OPERATIONS. Each operation is applied sequentially. For arrays, remove the highest indexed value first to avoid shifting indices. This ensures subsequent remove operations remain valid.3. add (for arrays, use /- to efficiently append to end)."
    ),
  patches: z
    .array(JsonPatchSchema)
    .describe(
      "Finally, provide a list of JSONPatch operations to be applied to the existing document. Take care to respect array bounds. Order patches as follows:\n 1. replace - this keeps collection size the same\n 2. remove - BE CAREFUL about order of operations. For arrays, remove the highest indexed value first to avoid shifting indices.\n 3. add - for arrays, use /- to efficiently append to end."
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
