import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  Annotation,
  StateGraph,
  START,
  END,
  Send,
} from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { zodToJsonSchema } from "zod-to-json-schema";

import type {
  ExistingType,
  SchemaInstance,
  ToolCall,
  ToolType,
} from "./types.js";
import { applyJsonPatches, ensurePatches } from "./json-patch.js";
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
 * Tuple message format: [role, content] or (role, content)
 * Matches Python's MessageLikeRepresentation tuple format.
 */
export type MessageTuple = [string, string];

/**
 * Union type for all message-like inputs.
 */
export type MessageLike = BaseMessage | MessageDict | MessageTuple | string;

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
 * Check if an object is a message tuple [role, content].
 */
function isMessageTuple(obj: unknown): obj is MessageTuple {
  return (
    Array.isArray(obj) &&
    obj.length === 2 &&
    typeof obj[0] === "string" &&
    typeof obj[1] === "string"
  );
}

/**
 * Check if an array contains message-like objects (dict or tuple format).
 */
function isMessageLikeArray(
  arr: unknown[]
): arr is Array<MessageDict | MessageTuple> {
  return arr.length > 0 && arr.every((item) => isMessageDict(item) || isMessageTuple(item));
}

/**
 * Convert a single message-like object to a BaseMessage.
 */
function convertMessageLike(msg: MessageDict | MessageTuple): BaseMessage {
  let role: string;
  let content: string;
  let toolCallId: string | undefined;
  let name: string | undefined;

  if (isMessageTuple(msg)) {
    [role, content] = msg;
  } else {
    role = msg.role;
    content = msg.content;
    toolCallId = msg.tool_call_id;
    name = msg.name;
  }

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
        tool_call_id: toolCallId || "",
        name,
      });
    default:
      return new HumanMessage({ content });
  }
}

/**
 * Convert an array of message-like objects to BaseMessage array.
 */
function convertMessageLikes(
  messages: Array<MessageDict | MessageTuple>
): BaseMessage[] {
  return messages.map(convertMessageLike);
}

/**
 * Extraction inputs type.
 */
export interface ExtractionInputs {
  messages: BaseMessage[] | MessageDict[] | MessageTuple[] | MessageLike[] | string;
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
    reducer: (curr, update) => {
      if (!curr) return update;
      return [...curr, ...update];
    },
    default: () => [],
  }),
  attempts: Annotation<number>({
    reducer: (curr, update) => (curr || 0) + update,
    default: () => 0,
  }),
  msgId: Annotation<string>({
    reducer: (curr, update) => curr || update,
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
function ensureTools(
  tools: ToolType[]
): Map<string, z.ZodSchema> {
  const result = new Map<string, z.ZodSchema>();

  for (const tool of tools) {
    if (tool instanceof z.ZodObject) {
      const name = tool.description || `Schema_${result.size}`;
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
export function createExtractor(
  llm: BaseChatModel,
  options: ExtractorOptions
) {
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
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    existingSchemaPolicy: _existingSchemaPolicy = true,
  } = options;

  // Convert tools to schemas
  const toolSchemas = ensureTools(tools);
  const toolNames = Array.from(toolSchemas.keys());

  // Add patch schemas
  toolSchemas.set("PatchDoc", PatchDocSchema);
  toolSchemas.set("PatchFunctionErrors", PatchFunctionErrorsSchema);

  // Create validation node
  const validator = new ValidationNode(
    Array.from(toolSchemas.entries()).map(([name, schema]) => {
      if (schema instanceof z.ZodObject) {
        return schema.describe(name);
      }
      return schema;
    }),
    {
      formatError: (error, call, schema) => {
        const jsonSchema = schema instanceof z.ZodObject 
          ? JSON.stringify(zodToJsonSchema(schema), null, 2)
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
    if (schema instanceof z.ZodObject) {
      return zodToOpenAIFunction(schema, name);
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
        if (schema instanceof z.ZodObject) {
          updateTools.push(zodToOpenAIFunction(schema, name));
        }
      }
    }

    // Build existing schemas context
    const schemaStrings: string[] = [];
    if (typeof existing === "object" && !Array.isArray(existing)) {
      for (const [k, v] of Object.entries(existing)) {
        const schema = toolSchemas.get(k);
        const schemaJson = schema instanceof z.ZodObject
          ? JSON.stringify(zodToJsonSchema(schema), null, 2)
          : "object";
        schemaStrings.push(
          `<schema id="${k}">\n<instance>\n${JSON.stringify(v, null, 2)}\n</instance>\n<json_schema>\n${schemaJson}\n</json_schema></schema>`
        );
      }
    } else if (Array.isArray(existing)) {
      for (const item of existing) {
        if (Array.isArray(item)) {
          const [id, typeName, record] = item;
          schemaStrings.push(
            `<instance id="${id}" schema_type="${typeName}">\n${JSON.stringify(record, null, 2)}\n</instance>`
          );
        } else {
          schemaStrings.push(
            `<instance id="${item.recordId}" schema_type="${item.schemaName}">\n${JSON.stringify(item.record, null, 2)}\n</instance>`
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
    if (enableDeletes && existing) {
      const existingIds = Array.isArray(existing)
        ? existing.map((e) =>
            Array.isArray(e) ? e[0] : (e as SchemaInstance).recordId
          )
        : Object.keys(existing);
      removalSchema = createRemoveDocSchema(existingIds);
      updateTools.push(
        zodToOpenAIFunction(removalSchema as z.ZodObject<z.ZodRawShape>, "RemoveDoc")
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

          if (typeof existing === "object" && !Array.isArray(existing)) {
            target = existing[jsonDocId] as Record<string, unknown>;
            toolName = jsonDocId;
          } else if (Array.isArray(existing)) {
            const found = existing.find((e) =>
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
      const response = await boundLlm.invoke(state.messages, config);
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
              if (msg instanceof AIMessage) {
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

      return {
        attempts: state.bumpAttempt ? 1 : 0,
      };
    } catch {
      return {};
    }
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

  // Validation routing
  function validateOrRetry(
    state: typeof ExtractionStateAnnotation.State
  ): "validate" | "extractUpdates" {
    const lastMsg = state.messages[state.messages.length - 1];
    if (lastMsg instanceof AIMessage) {
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

    const sends: Send[] = [];
    let bumped = false;

    // Check for validation errors
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const msg = state.messages[i];
      if (msg instanceof AIMessage) break;

      if (msg instanceof ToolMessage) {
        const isError = msg.additional_kwargs?.is_error;
        if (isError) {
          sends.push(
            new Send("patch", {
              ...state,
              toolCallId: msg.tool_call_id,
              bumpAttempt: !bumped,
            })
          );
          bumped = true;
        }
      }
    }

    return sends.length > 0 ? sends : END;
  }

  // Build the graph
  const builder = new StateGraph(ExtractionStateAnnotation)
    .addNode("extract", extract)
    .addNode("extractUpdates", extractUpdates)
    .addNode("validate", validate)
    .addNode("patch", patch)
    .addConditionalEdges(START, enter)
    .addEdge("extract", "validate")
    .addConditionalEdges("extractUpdates", validateOrRetry)
    .addConditionalEdges("validate", handleRetries, ["patch", END])
    .addEdge("patch", "validate");

  const compiled = builder.compile();

  // Create the runnable interface
  return {
    async invoke(
      input: ExtractionInputs | string | BaseMessage | MessageLike[],
      config?: RunnableConfig
    ): Promise<ExtractionOutputs> {
      // Coerce input to proper state type
      let messages: BaseMessage[];
      let existing: ExistingType | undefined;

      if (typeof input === "string") {
        messages = [new HumanMessage({ content: input })];
      } else if (input instanceof BaseMessage) {
        messages = [input];
      } else if (Array.isArray(input)) {
        // Check if it's an array of MessageDict/MessageTuple or BaseMessage
        if (isMessageLikeArray(input)) {
          messages = convertMessageLikes(input);
        } else {
          messages = input as BaseMessage[];
        }
      } else {
        // ExtractionInputs object
        if (typeof input.messages === "string") {
          messages = [new HumanMessage({ content: input.messages })];
        } else if (isMessageLikeArray(input.messages as unknown[])) {
          messages = convertMessageLikes(
            input.messages as Array<MessageDict | MessageTuple>
          );
        } else {
          messages = input.messages as BaseMessage[];
        }
        existing = input.existing;
      }

      const result = await compiled.invoke({ messages, existing }, config);

      // Filter and format output
      const msgId = result.msgId;
      const aiMessage = result.messages.find(
        (m: BaseMessage) => m.id === msgId && m instanceof AIMessage
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
        if (!schema || tc.name === "PatchDoc" || tc.name === "PatchFunctionErrors") {
          continue;
        }

        try {
          const validated = schema.parse(tc.args);
          responses.push(validated);
          responseMetadata.push({
            id: tc.id || "",
            jsonDocId: updatedDocs[tc.id || ""],
          });
        } catch (e) {
          console.error(`Validation failed for ${tc.name}:`, e);
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
      input: ExtractionInputs | string | BaseMessage | MessageLike[],
      config?: RunnableConfig
    ) {
      // Coerce input to proper state type
      let messages: BaseMessage[];
      let existing: ExistingType | undefined;

      if (typeof input === "string") {
        messages = [new HumanMessage({ content: input })];
      } else if (input instanceof BaseMessage) {
        messages = [input];
      } else if (Array.isArray(input)) {
        // Check if it's an array of MessageDict/MessageTuple or BaseMessage
        if (isMessageLikeArray(input)) {
          messages = convertMessageLikes(input);
        } else {
          messages = input as BaseMessage[];
        }
      } else {
        // ExtractionInputs object
        if (typeof input.messages === "string") {
          messages = [new HumanMessage({ content: input.messages })];
        } else if (isMessageLikeArray(input.messages as unknown[])) {
          messages = convertMessageLikes(
            input.messages as Array<MessageDict | MessageTuple>
          );
        } else {
          messages = input.messages as BaseMessage[];
        }
        existing = input.existing;
      }

      return compiled.stream({ messages, existing }, config);
    },
  };
}