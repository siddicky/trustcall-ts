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
import { isZodSchema, getSchemaName } from "./types.js";

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
      if (isZodSchema(schema)) {
        // Any Zod schema - use the description or a generated name
        const name = getSchemaName(schema, `Schema_${this.schemasByName.size}`);
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

  private jsonSchemaToZod(schema: Record<string, unknown>): z.ZodSchema {
    return this._convertJsonSchema(schema);
  }

  /**
   * Recursively convert a JSON Schema object to a Zod schema.
   * Handles common types: string, number, integer, boolean, null, array, object, enum.
   * Falls back to z.unknown() for unsupported features.
   */
  private _convertJsonSchema(schema: Record<string, unknown>): z.ZodSchema {
    if (!schema || typeof schema !== 'object') {
      return z.unknown();
    }

    // Handle enum
    if (Array.isArray(schema.enum)) {
      const values = schema.enum as [string, ...string[]];
      if (values.length > 0 && values.every((v) => typeof v === 'string')) {
        return z.enum(values as [string, ...string[]]);
      }
      const literals = values.map((v) => z.literal(v as string | number | boolean));
      if (literals.length >= 2) {
        return z.union(literals as unknown as [z.ZodTypeAny, z.ZodTypeAny, ...z.ZodTypeAny[]]);
      }
      return literals[0] ?? z.unknown();
    }

    // Handle const
    if ('const' in schema) {
      return z.literal(schema.const as string | number | boolean);
    }

    const type = schema.type as string | undefined;

    switch (type) {
      case 'string':
        return z.string();
      case 'number':
      case 'integer':
        return z.number();
      case 'boolean':
        return z.boolean();
      case 'null':
        return z.null();
      case 'array': {
        const items = schema.items as Record<string, unknown> | undefined;
        if (items) {
          return z.array(this._convertJsonSchema(items));
        }
        return z.array(z.unknown());
      }
      case 'object': {
        const properties = schema.properties as Record<string, Record<string, unknown>> | undefined;
        const required = (schema.required as string[]) || [];
        if (properties) {
          const shape: Record<string, z.ZodTypeAny> = {};
          for (const [key, propSchema] of Object.entries(properties)) {
            const converted = this._convertJsonSchema(propSchema);
            shape[key] = required.includes(key) ? converted : converted.optional();
          }
          if (schema.additionalProperties !== false) {
            return z.object(shape).passthrough();
          }
          return z.object(shape);
        }
        return z.object({}).passthrough();
      }
      default:
        // No type specified or unsupported type
        if (schema.properties) {
          // Treat as object if it has properties
          return this._convertJsonSchema({ ...schema, type: 'object' });
        }
        return z.unknown();
    }
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
