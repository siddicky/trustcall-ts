import { z } from "zod";
import type { BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";

/**
 * Represents an instance of a schema with its associated metadata.
 */
export interface SchemaInstance {
  /** A unique identifier for this schema instance */
  recordId: string;
  /** The name of the schema that this instance conforms to */
  schemaName: string;
  /** The actual data of the record */
  record: Record<string, unknown>;
}

/**
 * Type for existing schemas that can be updated.
 */
export type ExistingType =
  | Record<string, unknown>
  | SchemaInstance[]
  | Array<[string, string, Record<string, unknown>]>;

/**
 * Tool types that can be used with the extractor.
 */
export type ToolType =
  | z.ZodObject<z.ZodRawShape>
  | z.ZodSchema
  | StructuredToolInterface
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ((...args: any[]) => any)
  | {
      name: string;
      description?: string;
      parameters: Record<string, unknown>;
    };

/**
 * JSON Patch operation types based on RFC 6902.
 */
export interface JsonPatchOp {
  op: "add" | "remove" | "replace" | "move" | "copy" | "test";
  path: string;
  value?: unknown;
  from?: string;
}

/**
 * Tool call interface for LLM responses.
 */
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/**
 * Internal extraction state.
 */
export interface ExtractionState {
  messages: BaseMessage[];
  attempts: number;
  msgId: string;
  existing?: ExistingType;
}

/**
 * Extended extraction state for patching operations.
 */
export interface ExtendedExtractState extends ExtractionState {
  toolCallId: string;
  bumpAttempt: boolean;
}

/**
 * Deletion state for removing tool calls.
 */
export interface DeletionState extends ExtractionState {
  deletionTarget: string;
}

/**
 * Message operation types for state updates.
 */
export interface MessageOp {
  op: "delete" | "update_tool_call" | "update_tool_name";
  target: string | ToolCall;
}