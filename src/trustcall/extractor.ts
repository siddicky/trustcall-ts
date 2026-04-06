import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  isAIMessage,
  isBaseMessage,
} from "@langchain/core/messages";
import { Annotation, StateGraph, START, END, Send } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { zodToJsonSchema } from "zod-to-json-schema";

import type {
  ExistingType,
  MessageOp,
  SchemaInstance,
  ToolCall,
  ToolType,
} from "./types.js";
import { isZodSchema, getSchemaName } from "./types.js";
import { applyJsonPatches, ensurePatches } from "./json-patch.js";
import { applyMessageOps, getHistoryForToolCall } from "./utils.js";
import {
  PatchDocSchema,
  PatchFunctionErrorsSchema,
  createPatchFunctionNameSchema,
  createRemoveDocSchema,
} from "./schemas.js";
import { ValidationNode } from "./validation-node.js";

const DEFAULT_MAX_ATTEMPTS = 3;

/**
 * OpenAI-style message dictionary format.
 */
export interface MessageDict {
  role: "system" | "user" | "assistant" | "tool" | "human" | "ai";
  content: string;
  tool_call_id?: string;
  name?: string;
}

/**
 * Check if an object is a MessageDict (OpenAI-style message).
 */
function isMessageDict(obj: unknown): obj is MessageDict {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "role" in obj &&
    "content" in obj &&
    typeof (obj as MessageDict).role === "string" &&
    ["system", "user", "assistant", "tool", "human", "ai"].includes(
      (obj as MessageDict).role
    )
  );
}

/**
 * Check if an array contains MessageDict objects (OpenAI-style messages).
 */
function isMessageDictArray(arr: unknown[]): arr is MessageDict[] {
  return arr.length > 0 && arr.every((item) => isMessageDict(item));
}

/**
 * Check if an array contains BaseMessage objects.
 */
function isBaseMessageArray(arr: unknown[]): arr is BaseMessage[] {
  return arr.length > 0 && arr.every((item) => isBaseMessage(item));
}

/**
 * Convert a MessageDict (OpenAI-style) to a BaseMessage.
 */
function convertMessageDict(msg: MessageDict): BaseMessage {
  const { role, content, tool_call_id, name } = msg;

  switch (role) {
    case "system":
      return new SystemMessage({ content });
    case "user":
    case "human":
      return new HumanMessage({ content });
    case "assistant":
    case "ai":
      return new AIMessage({ content });
    case "tool":
      return new ToolMessage({
        content,
        tool_call_id: tool_call_id || "",
        name,
      });
    default:
      return new HumanMessage({ content });
  }
}

/**
 * Convert an array of MessageDict objects to BaseMessage array.
 */
function convertMessageDicts(messages: MessageDict[]): BaseMessage[] {
  return messages.map(convertMessageDict);
}

/**
 * Extraction inputs type.
 *
 * The `messages` field supports multiple formats:
 * - A string (converted to HumanMessage internally)
 * - Array of BaseMessage instances (LangGraph MessagesValue compatible)
 * - Array of MessageDict (OpenAI-style { role, content })
 *
 * For simple extraction, you can also call invoke() directly with:
 * - A string
 * - A single BaseMessage
 */
export interface ExtractionInputs {
  messages: string | BaseMessage[] | MessageDict[];
  existing?: ExistingType;
}

/**
 * Extraction outputs type.
 */
export interface ExtractionOutputs {
  messages: AIMessage[];
  responses: z.infer<z.ZodSchema>[];
  responseMetadata: Array<{ id: string; jsonDocId?: string }>;
  attempts: number;
}

/**
 * Options for creating an extractor.
 */
export interface ExtractorOptions {
  /** The tools/schemas to extract */
  tools: ToolType[];
  /** Specific tool to force usage of */
  toolChoice?: string;
  /** Allow inserting new schemas when updating */
  enableInserts?: boolean;
  /** Allow updating existing schemas */
  enableUpdates?: boolean;
  /** Allow deleting existing schemas */
  enableDeletes?: boolean;
  /**
   * Policy for handling existing schemas that don't match provided tools.
   * true = raise error, false = treat as dict, "ignore" = drop
   */
  existingSchemaPolicy?: boolean | "ignore";
}

// State annotation for the extraction graph
const ExtractionStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (curr: BaseMessage[] | undefined, update: any) => {
      const currentMessages = curr ?? [];
      const nextItems = Array.isArray(update) ? update : [];
      const nextMessages: BaseMessage[] = [];
      const messageOps: MessageOp[] = [];

      for (const item of nextItems) {
        if (isBaseMessage(item)) {
          nextMessages.push(item);
          continue;
        }

        if (
          item &&
          typeof item === "object" &&
          "op" in item &&
          "target" in item
        ) {
          messageOps.push(item as MessageOp);
        }
      }

      const combinedMessages = [...currentMessages, ...nextMessages];
      if (messageOps.length === 0) {
        return combinedMessages;
      }

      return applyMessageOps(combinedMessages, messageOps);
    },
    default: () => [],
  }),
  attempts: Annotation<number>({
    reducer: (curr: number | undefined, update: number) => (curr || 0) + update,
    default: () => 0,
  }),
  msgId: Annotation<string>({
    reducer: (curr: string | undefined, update: string) => curr || update,
    default: () => "",
  }),
  existing: Annotation<ExistingType | undefined>,
  toolCallId: Annotation<string>,
  bumpAttempt: Annotation<boolean>,
});

/**
 * Convert a Zod schema to OpenAI function format.
 */
function zodToOpenAIFunction(schema: z.ZodObject<z.ZodRawShape>, name: string) {
  const jsonSchema = zodToJsonSchema(schema, { target: "openAi" });
  return {
    type: "function" as const,
    function: {
      name,
      description: schema.description || "",
      parameters: jsonSchema,
    },
  };
}


/**
 * Convert tools to a standardized format.
 */
function ensureTools(tools: ToolType[]): Map<string, z.ZodSchema> {
  const result = new Map<string, z.ZodSchema>();

  for (const tool of tools) {
    if (isZodSchema(tool)) {
      // Any Zod schema - use description or generated name
      const name = getSchemaName(tool, `Schema_${result.size}`);
      result.set(name, tool);
    } else if (typeof tool === "object" && "name" in tool) {
      // Already in correct format or structured tool
      if ("schema" in tool && tool.schema) {
        result.set(tool.name, tool.schema as z.ZodSchema);
      } else if ("parameters" in tool) {
        // Convert parameters to passthrough schema
        result.set(tool.name, z.object({}).passthrough());
      }
    } else if (typeof tool === "function") {
      result.set(tool.name, z.object({}).passthrough());
    }
  }

  return result;
}

/**
 * Create an extractor that generates validated structured outputs using an LLM.
 *
 * This function binds validators and retry logic to ensure the validity of
 * generated tool calls. It uses JSONPatch to correct validation errors caused
 * by incorrect or incomplete parameters in previous tool calls.
 *
 * @example
 * ```typescript
 * import { z } from "zod";
 * import { ChatOpenAI } from "@langchain/openai";
 * import { createExtractor } from "trustcalljs";
 *
 * const UserInfo = z.object({
 *   name: z.string().describe("User's full name"),
 *   age: z.number().describe("User's age in years"),
 * }).describe("UserInfo");
 *
 * const llm = new ChatOpenAI({ model: "gpt-4" });
 *
 * const extractor = createExtractor(llm, {
 *   tools: [UserInfo],
 * });
 *
 * const result = await extractor.invoke({
 *   messages: "My name is Alice and I'm 30 years old",
 * });
 *
 * console.log(result.responses[0]);
 * // { name: "Alice", age: 30 }
 * ```
 */
export function createExtractor(llm: BaseChatModel, options: ExtractorOptions) {
  // Verify the LLM supports tool binding
  if (!llm.bindTools) {
    throw new Error(
      "The provided LLM does not support tool binding. " +
        "Please use a model that supports tool calling (e.g., ChatOpenAI, ChatAnthropic)."
    );
  }

  // Cast to a type that has bindTools defined
  const toolLlm = llm as BaseChatModel & {
    bindTools: NonNullable<BaseChatModel["bindTools"]>;
  };

  const {
    tools,
    toolChoice,
    enableInserts = false,
    enableUpdates = true,
    enableDeletes = false,
    existingSchemaPolicy = true,
  } = options;

  // Convert tools to schemas
  const toolSchemas = ensureTools(tools);
  const toolNames = Array.from(toolSchemas.keys());

  // Add patch schemas
  toolSchemas.set("PatchDoc", PatchDocSchema);
  toolSchemas.set("PatchFunctionErrors", PatchFunctionErrorsSchema);

  // Register RemoveDoc as a passthrough since it's dynamically created per-invocation
  if (enableDeletes) {
    toolSchemas.set("RemoveDoc", z.object({ json_doc_id: z.string() }).passthrough());
  }

  /**
   * Validate existing data against known schemas based on existingSchemaPolicy.
   * - true (default): throw on unknown schemas
   * - false: treat unknown schemas as generic dicts (passthrough)
   * - "ignore": silently drop unknown schemas
   */
  function validateExisting(existing: ExistingType): ExistingType {
    if (existingSchemaPolicy === "ignore" || existingSchemaPolicy === false) {
      if (typeof existing === "object" && !Array.isArray(existing)) {
        const validated: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(existing)) {
          if (toolNames.includes(key) || key === "__any__") {
            validated[key] = value;
          } else if (existingSchemaPolicy === false) {
            validated[key] = value; // Keep as generic dict
          }
          // "ignore" mode: silently skip unknown keys
        }
        return validated;
      } else if (Array.isArray(existing)) {
        return existing.filter((item: SchemaInstance | [string, string, Record<string, unknown>]) => {
          const schemaName = Array.isArray(item) ? item[1] : (item as SchemaInstance).schemaName;
          if (toolNames.includes(schemaName) || schemaName === "__any__") return true;
          if (existingSchemaPolicy === false) return true;
          return false; // "ignore" mode: drop unknown
        }) as ExistingType;
      }
    } else {
      // true (default): throw on unknown schemas
      if (typeof existing === "object" && !Array.isArray(existing)) {
        for (const key of Object.keys(existing)) {
          if (!toolNames.includes(key) && key !== "__any__") {
            throw new Error(
              `Key '${key}' doesn't match any schema. Known schemas: ${toolNames.join(", ")}`
            );
          }
        }
      } else if (Array.isArray(existing)) {
        for (let i = 0; i < existing.length; i++) {
          const item = existing[i];
          const schemaName = Array.isArray(item) ? item[1] : (item as SchemaInstance).schemaName;
          if (!toolNames.includes(schemaName) && schemaName !== "__any__") {
            throw new Error(
              `Unknown schema '${schemaName}' at index ${i}. Known schemas: ${toolNames.join(", ")}`
            );
          }
        }
      }
    }
    return existing;
  }
  // Create validation node
  const validator = new ValidationNode(
    Array.from(toolSchemas.entries()).map(([name, schema]) => {
      if (isZodSchema(schema)) {
        return schema.describe(name);
      }
      return schema;
    }),
    {
      formatError: (error, call, schema) => {
        const jsonSchema = isZodSchema(schema)
          ? JSON.stringify(zodToJsonSchema(schema as z.ZodSchema), null, 2)
          : "{}";
        return (
          `Error:\n\n\`\`\`\n${error.message}\n\`\`\`\n` +
          `Expected Parameter Schema:\n\n\`\`\`json\n${jsonSchema}\n\`\`\`\n` +
          `Please use PatchFunctionErrors to fix all validation errors ` +
          `for json_doc_id=[${call.id}].`
        );
      },
    }
  );

  // Build the extraction tools for LLM binding
  const extractionTools = toolNames.map((name) => {
    const schema = toolSchemas.get(name);
    if (isZodSchema(schema)) {
      return zodToOpenAIFunction(schema as z.ZodObject<z.ZodRawShape>, name);
    }
    return {
      type: "function" as const,
      function: {
        name,
        description: "",
        parameters: {},
      },
    };
  });

  // Extract node - initial extraction without existing data
  async function extract(
    state: typeof ExtractionStateAnnotation.State,
    config: RunnableConfig
  ): Promise<Partial<typeof ExtractionStateAnnotation.State>> {
    const boundLlm = toolLlm.bindTools(extractionTools, {
      tool_choice: toolChoice,
    });

    const response = await boundLlm.invoke(state.messages, config);
    const aiMessage = response as AIMessage;

    if (!aiMessage.id) {
      aiMessage.id = uuidv4();
    }

    return {
      messages: [aiMessage],
      attempts: 1,
      msgId: aiMessage.id,
    };
  }

  // Extract updates node - for updating existing schemas
  async function extractUpdates(
    state: typeof ExtractionStateAnnotation.State,
    config: RunnableConfig
  ): Promise<Partial<typeof ExtractionStateAnnotation.State>> {
    const existing = state.existing;
    if (!existing) {
      throw new Error("No existing schemas provided.");
    }

    // Validate existing data against known schemas
    const validatedExisting = validateExisting(existing);

    // Build the update tools
    const updateTools: Array<{
      type: "function";
      function: { name: string; description: string; parameters: unknown };
    }> = [];

    if (enableUpdates) {
      updateTools.push(zodToOpenAIFunction(PatchDocSchema, "PatchDoc"));
    }

    if (enableInserts) {
      for (const name of toolNames) {
        const schema = toolSchemas.get(name);
        if (isZodSchema(schema)) {
          updateTools.push(
            zodToOpenAIFunction(schema as z.ZodObject<z.ZodRawShape>, name)
          );
        }
      }
    }

    // Build existing schemas context
    const schemaStrings: string[] = [];
    if (typeof validatedExisting === "object" && !Array.isArray(validatedExisting)) {
      for (const [k, v] of Object.entries(validatedExisting)) {
        const schema = toolSchemas.get(k);
        const schemaJson = isZodSchema(schema)
          ? JSON.stringify(zodToJsonSchema(schema as z.ZodSchema), null, 2)
          : "object";
        schemaStrings.push(
          `<schema id="${k}">\n<instance>\n${JSON.stringify(v, null, 2)}\n</instance>\n<json_schema>\n${schemaJson}\n</json_schema></schema>`
        );
      }
    } else if (Array.isArray(validatedExisting)) {
      for (const item of validatedExisting) {
        if (Array.isArray(item)) {
          const [id, typeName, record] = item;
          const schema = toolSchemas.get(typeName);
          const schemaJson = isZodSchema(schema)
            ? JSON.stringify(zodToJsonSchema(schema as z.ZodSchema), null, 2)
            : "object";
          schemaStrings.push(
            `<instance id="${id}" schema_type="${typeName}">\n${JSON.stringify(record, null, 2)}\n</instance>\n<json_schema>\n${schemaJson}\n</json_schema>`
          );
        } else {
          const schema = toolSchemas.get(item.schemaName);
          const schemaJson = isZodSchema(schema)
            ? JSON.stringify(zodToJsonSchema(schema as z.ZodSchema), null, 2)
            : "object";
          schemaStrings.push(
            `<instance id="${item.recordId}" schema_type="${item.schemaName}">\n${JSON.stringify(item.record, null, 2)}\n</instance>\n<json_schema>\n${schemaJson}\n</json_schema>`
          );
        }
      }
    }

    const existingMsg = `Generate JSONPatches to update the existing schema instances.${
      enableInserts
        ? " If you need to extract or insert *new* instances, call the relevant function(s)."
        : ""
    }
<existing>
${schemaStrings.join("\n")}
</existing>`;

    // Prepare messages with system context
    let messages = [...state.messages];
    if (messages[0] instanceof SystemMessage) {
      const sysMsg = messages[0] as SystemMessage;
      messages[0] = new SystemMessage({
        content: `${sysMsg.content}\n\n${existingMsg}`,
      });
    } else {
      messages = [new SystemMessage({ content: existingMsg }), ...messages];
    }

    // Handle deletions
    let removalSchema: z.ZodSchema | undefined;
    if (enableDeletes && validatedExisting) {
      const existingIds = Array.isArray(validatedExisting)
        ? validatedExisting.map((e) =>
            Array.isArray(e) ? e[0] : (e as SchemaInstance).recordId
          )
        : Object.keys(validatedExisting);
      removalSchema = createRemoveDocSchema(existingIds);
      updateTools.push(
        zodToOpenAIFunction(
          removalSchema as z.ZodObject<z.ZodRawShape>,
          "RemoveDoc"
        )
      );
    }

    const boundLlm = toolLlm.bindTools(updateTools, {
      tool_choice: enableDeletes ? "any" : "PatchDoc",
    });

    try {
      const response = await boundLlm.invoke(messages, config);
      const aiMessage = response as AIMessage;

      // Process tool calls and apply patches
      const resolvedToolCalls: ToolCall[] = [];
      const updatedDocs: Record<string, string> = {};

      for (const tc of aiMessage.tool_calls || []) {
        if (tc.name === "PatchDoc") {
          const args = tc.args as Record<string, unknown>;
          const jsonDocId = args.json_doc_id as string;

          let target: Record<string, unknown> | undefined;
          let toolName: string;

          if (typeof validatedExisting === "object" && !Array.isArray(validatedExisting)) {
            target = validatedExisting[jsonDocId] as Record<string, unknown>;
            toolName = jsonDocId;
          } else if (Array.isArray(validatedExisting)) {
            const found = validatedExisting.find((e) =>
              Array.isArray(e) ? e[0] === jsonDocId : e.recordId === jsonDocId
            );
            if (found) {
              if (Array.isArray(found)) {
                toolName = found[1];
                target = found[2];
              } else {
                toolName = found.schemaName;
                target = found.record;
              }
            } else {
              continue;
            }
          } else {
            continue;
          }

          if (target) {
            const patches = ensurePatches(args);
            if (patches.length > 0) {
              const patched = applyJsonPatches(target, patches);
              resolvedToolCalls.push({
                id: tc.id || uuidv4(),
                name: toolName,
                args: patched,
              });
              updatedDocs[tc.id || ""] = jsonDocId;
            }
          }
        } else {
          resolvedToolCalls.push({
            id: tc.id || uuidv4(),
            name: tc.name,
            args: tc.args as Record<string, unknown>,
          });
        }
      }

      const resultMessage = new AIMessage({
        content: aiMessage.content,
        tool_calls: resolvedToolCalls.map((tc) => ({
          id: tc.id,
          name: tc.name,
          args: tc.args,
        })),
        additional_kwargs: { updated_docs: updatedDocs },
      });

      if (!resultMessage.id) {
        resultMessage.id = uuidv4();
      }

      return {
        messages: [resultMessage],
        attempts: 1,
        msgId: resultMessage.id,
      };
    } catch (e) {
      return {
        messages: [
          new HumanMessage({
            content: `Fix the validation error while also avoiding: ${String(e)}`,
          }),
        ],
        attempts: 1,
      };
    }
  }

  // Validate node
  async function validate(
    state: typeof ExtractionStateAnnotation.State,
    config: RunnableConfig
  ): Promise<{ messages: ToolMessage[] }> {
    const result = await validator.invoke({ messages: state.messages }, config);
    return result as { messages: ToolMessage[] };
  }

  // Patch node - fix validation errors
  async function patch(
    state: typeof ExtractionStateAnnotation.State,
    config: RunnableConfig
  ): Promise<Partial<typeof ExtractionStateAnnotation.State>> {
    const patchTools = [
      zodToOpenAIFunction(PatchFunctionErrorsSchema, "PatchFunctionErrors"),
      zodToOpenAIFunction(
        createPatchFunctionNameSchema(toolNames) as z.ZodObject<z.ZodRawShape>,
        "PatchFunctionName"
      ),
    ];

    const boundLlm = toolLlm.bindTools(patchTools, { tool_choice: "any" });

    try {
      const filteredMessages = getHistoryForToolCall(state.messages, state.toolCallId);
      const response = await boundLlm.invoke(filteredMessages, config);
      const aiMessage = response as AIMessage;

      // Apply patches to fix errors
      const messageOps: Array<{
        op: string;
        target: unknown;
      }> = [];

      for (const tc of aiMessage.tool_calls || []) {
        const args = tc.args as Record<string, unknown>;
        const targetId = state.toolCallId;

        if (tc.name === "PatchFunctionName" && args.fixed_name) {
          messageOps.push({
            op: "update_tool_name",
            target: { id: targetId, name: args.fixed_name },
          });
        } else if (tc.name === "PatchFunctionErrors") {
          const patches = ensurePatches(args);
          if (patches.length > 0) {
            // Find original tool call and apply patches
            for (const msg of state.messages) {
              if (isAIMessage(msg)) {
                for (const origTc of msg.tool_calls || []) {
                  if (origTc.id === targetId) {
                    const patchedArgs = applyJsonPatches(
                      origTc.args as Record<string, unknown>,
                      patches
                    );
                    messageOps.push({
                      op: "update_tool_call",
                      target: {
                        id: targetId,
                        name: origTc.name,
                        args: patchedArgs,
                      },
                    });
                  }
                }
              }
            }
          }
        }
      }

      messageOps.push({
        op: "delete",
        target: state.toolCallId,
      });

      return {
        messages: [aiMessage, ...messageOps] as any,
        attempts: state.bumpAttempt ? 1 : 0,
      };
    } catch {
      return {};
    }
  }

  // Delete tool call node - removes successful validation ToolMessages
  function delToolCall(
    state: typeof ExtractionStateAnnotation.State
  ): Partial<typeof ExtractionStateAnnotation.State> {
    return {
      messages: [{ op: "delete", target: state.toolCallId }] as any,
    };
  }

  // Entry routing
  function enter(
    state: typeof ExtractionStateAnnotation.State
  ): "extract" | "extractUpdates" {
    if (state.existing) {
      return "extractUpdates";
    }
    return "extract";
  }

  function findLastAiMessage(messages: BaseMessage[]): AIMessage | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (isAIMessage(msg)) {
        return msg;
      }
    }
    return undefined;
  }

  // Validation routing
  function validateOrRetry(
    state: typeof ExtractionStateAnnotation.State
  ): "validate" | "extractUpdates" {
    const lastMsg = findLastAiMessage(state.messages);
    if (lastMsg) {
      return "validate";
    }
    return "extractUpdates";
  }

  // Handle retries after validation
  function handleRetries(
    state: typeof ExtractionStateAnnotation.State,
    config: RunnableConfig
  ): typeof END | Send[] {
    const maxAttempts =
      (config.configurable?.max_attempts as number) || DEFAULT_MAX_ATTEMPTS;

    if (state.attempts >= maxAttempts) {
      return END;
    }

    const patchSends: Send[] = [];
    const delSends: Send[] = [];
    let bumped = false;

    // Check for validation errors
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const msg = state.messages[i];
      if (isAIMessage(msg)) break;

      if (msg instanceof ToolMessage) {
        const isError = msg.additional_kwargs?.is_error;
        if (isError) {
          patchSends.push(
            new Send("patch", {
              ...state,
              toolCallId: msg.tool_call_id,
              bumpAttempt: !bumped,
            })
          );
          bumped = true;
        } else {
          // Queue successful validation ToolMessages for deletion
          // to avoid mixing branches during fan-in
          delSends.push(
            new Send("delToolCall", {
              ...state,
              toolCallId: String(msg.id),
            })
          );
        }
      }
    }

    // Only send delToolCall when we also have patches to apply.
    // If all validations passed (no errors), just END.
    if (patchSends.length > 0) {
      return [...patchSends, ...delSends];
    }

    return END;
  }

  // Build the graph
  const builder = new StateGraph(ExtractionStateAnnotation)
    .addNode("extract", extract)
    .addNode("extractUpdates", extractUpdates)
    .addNode("validate", validate)
    .addNode("patch", patch)
    .addNode("delToolCall", delToolCall)
    .addConditionalEdges(START, enter)
    .addEdge("extract", "validate")
    .addConditionalEdges("extractUpdates", validateOrRetry)
    .addConditionalEdges("validate", handleRetries, ["patch", "delToolCall", END])
    .addEdge("patch", "validate")
    .addEdge("delToolCall", "validate");

  const compiled = builder.compile();

  // Create the runnable interface
  return {
    async invoke(
      input: ExtractionInputs | string | BaseMessage,
      config?: RunnableConfig
    ): Promise<ExtractionOutputs> {
      // Coerce input to proper state type
      let messages: BaseMessage[];
      let existing: ExistingType | undefined;

      if (typeof input === "string") {
        // Simple string input
        messages = [new HumanMessage({ content: input })];
      } else if (isBaseMessage(input)) {
        // Single BaseMessage input
        messages = [input];
      } else {
        // ExtractionInputs object with { messages: ..., existing?: ... }
        // Supports string, array of BaseMessage, or array of MessageDict
        if (typeof input.messages === "string") {
          messages = [new HumanMessage({ content: input.messages })];
        } else if (
          Array.isArray(input.messages) &&
          isBaseMessageArray(input.messages)
        ) {
          messages = input.messages;
        } else if (
          Array.isArray(input.messages) &&
          isMessageDictArray(input.messages)
        ) {
          messages = convertMessageDicts(input.messages);
        } else {
          messages = input.messages as BaseMessage[];
        }
        existing = input.existing;
      }

      const result = await compiled.invoke({ messages, existing }, config);

      // Filter and format output
      const msgId = result.msgId;
      const aiMessage = result.messages.find(
        (m: BaseMessage) => m.id === msgId && isAIMessage(m)
      ) as AIMessage | undefined;

      if (!aiMessage) {
        return {
          messages: [],
          responses: [],
          responseMetadata: [],
          attempts: result.attempts,
        };
      }

      const responses: z.infer<z.ZodSchema>[] = [];
      const responseMetadata: Array<{ id: string; jsonDocId?: string }> = [];
      const updatedDocs =
        (aiMessage.additional_kwargs?.updated_docs as Record<string, string>) ||
        {};

      for (const tc of aiMessage.tool_calls || []) {
        const schema = toolSchemas.get(tc.name);
        if (
          !schema ||
          tc.name === "PatchDoc" ||
          tc.name === "PatchFunctionErrors"
        ) {
          continue;
        }

        try {
          const validated = schema.parse(tc.args);
          responses.push(validated);
          responseMetadata.push({
            id: tc.id || "",
            jsonDocId: updatedDocs[tc.id || ""],
          });
        } catch {
          // Silently skip tool calls that fail final validation
        }
      }

      return {
        messages: [aiMessage],
        responses,
        responseMetadata,
        attempts: result.attempts,
      };
    },

    async stream(
      input: ExtractionInputs | string | BaseMessage,
      config?: RunnableConfig
    ) {
      // Coerce input to proper state type
      let messages: BaseMessage[];
      let existing: ExistingType | undefined;

      if (typeof input === "string") {
        // Simple string input
        messages = [new HumanMessage({ content: input })];
      } else if (isBaseMessage(input)) {
        // Single BaseMessage input
        messages = [input];
      } else {
        // ExtractionInputs object with { messages: ..., existing?: ... }
        // Supports string, array of BaseMessage, or array of MessageDict
        if (typeof input.messages === "string") {
          messages = [new HumanMessage({ content: input.messages })];
        } else if (
          Array.isArray(input.messages) &&
          isBaseMessageArray(input.messages)
        ) {
          messages = input.messages;
        } else if (
          Array.isArray(input.messages) &&
          isMessageDictArray(input.messages)
        ) {
          messages = convertMessageDicts(input.messages);
        } else {
          messages = input.messages as BaseMessage[];
        }
        existing = input.existing;
      }

      return compiled.stream({ messages, existing }, config);
    },
  };
}
