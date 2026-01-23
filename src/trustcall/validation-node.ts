import { z } from "zod";
import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  isAIMessage,
} from "@langchain/core/messages";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { ToolCall, ToolType } from "./types.js";

export interface ValidationNodeOptions {
  /** Custom error formatter */
  formatError?: (error: Error, call: ToolCall, schema: z.ZodSchema) => string;
  /** Node name for tracing */
  name?: string;
  /** Tags for tracing */
  tags?: string[];
}

/**
 * Default error formatter.
 */
function defaultFormatError(
  error: Error,
  _call: ToolCall,
  _schema: z.ZodSchema
): string {
  return `${error.message}\n\nRespond after fixing all validation errors.`;
}

/**
 * A node that validates all tool calls from the last AIMessage.
 *
 * This node does not actually run the tools, it only validates the tool calls,
 * which is useful for extraction and other use cases where you need to generate
 * structured output that conforms to a complex schema.
 */
export class ValidationNode {
  public schemasByName: Map<string, z.ZodSchema>;
  private formatError: (
    error: Error,
    call: ToolCall,
    schema: z.ZodSchema
  ) => string;
  public name: string;
  public tags: string[];

  constructor(schemas: ToolType[], options: ValidationNodeOptions = {}) {
    this.formatError = options.formatError || defaultFormatError;
    this.name = options.name || "validation";
    this.tags = options.tags || [];
    this.schemasByName = new Map();

    for (const schema of schemas) {
      if (schema instanceof z.ZodObject) {
        // Zod schema - use the description or a generated name
        const name =
          (schema.description as string) || `Schema_${this.schemasByName.size}`;
        this.schemasByName.set(name, schema);
      } else if (this.isStructuredTool(schema)) {
        // Structured tool
        if (schema.schema) {
          this.schemasByName.set(schema.name, schema.schema as z.ZodSchema);
        }
      } else if (typeof schema === "function") {
        // Function - create schema from function
        this.schemasByName.set(schema.name, z.object({}).passthrough());
      } else if (typeof schema === "object" && "name" in schema) {
        // Dict with name, description, parameters
        const zodSchema = this.jsonSchemaToZod(schema.parameters);
        this.schemasByName.set(schema.name, zodSchema);
      }
    }
  }

  private isStructuredTool(obj: unknown): obj is StructuredToolInterface {
    return (
      typeof obj === "object" &&
      obj !== null &&
      "name" in obj &&
      "schema" in obj
    );
  }

  private jsonSchemaToZod(_schema: Record<string, unknown>): z.ZodSchema {
    // Simplified JSON Schema to Zod conversion
    // In production, you'd want a more robust implementation
    return z.object({}).passthrough();
  }

  /**
   * Get the last AIMessage from input.
   */
  private getMessage(input: BaseMessage[] | { messages: BaseMessage[] }): {
    outputType: "list" | "dict";
    message: AIMessage;
  } {
    let messages: BaseMessage[];
    let outputType: "list" | "dict";

    if (Array.isArray(input)) {
      messages = input;
      outputType = "list";
    } else {
      messages = input.messages || [];
      outputType = "dict";
    }

    const lastMessage = messages[messages.length - 1];
    if (!isAIMessage(lastMessage)) {
      throw new Error("Last message is not an AIMessage");
    }

    return { outputType, message: lastMessage as AIMessage };
  }

  /**
   * Validate a single tool call.
   */
  private validateOne(call: {
    id?: string;
    name: string;
    args: Record<string, unknown>;
  }): ToolMessage {
    const toolCall: ToolCall = {
      id: call.id || "",
      name: call.name,
      args: call.args as Record<string, unknown>,
    };

    const schema = this.schemasByName.get(toolCall.name);

    if (!schema) {
      const validNames = Array.from(this.schemasByName.keys()).join(", ");
      return new ToolMessage({
        content: `Unrecognized tool name: "${toolCall.name}". You only have access to the following tools: ${validNames}. Please call PatchFunctionName with the *correct* tool name to fix json_doc_id=[${toolCall.id}].`,
        tool_call_id: toolCall.id,
        name: toolCall.name,
        status: "error",
        additional_kwargs: { is_error: true },
      });
    }

    try {
      const result = schema.parse(toolCall.args);
      return new ToolMessage({
        content: JSON.stringify(result),
        tool_call_id: toolCall.id,
        name: toolCall.name,
        status: "success",
      });
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      return new ToolMessage({
        content: this.formatError(error, toolCall, schema),
        tool_call_id: toolCall.id,
        name: toolCall.name,
        status: "error",
        additional_kwargs: { is_error: true },
      });
    }
  }

  /**
   * Validate and run tool calls in parallel.
   */
  async invoke(
    input: BaseMessage[] | { messages: BaseMessage[] },
    _config?: RunnableConfig
  ): Promise<ToolMessage[] | { messages: ToolMessage[] }> {
    const { outputType, message } = this.getMessage(input);

    const toolCalls = message.tool_calls || [];

    // Run validations in parallel
    const outputs = await Promise.all(
      toolCalls.map((call) =>
        Promise.resolve(
          this.validateOne({
            id: call.id,
            name: call.name,
            args: call.args as Record<string, unknown>,
          })
        )
      )
    );

    return outputType === "list" ? outputs : { messages: outputs };
  }
}
