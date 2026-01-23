import { AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import type { ToolCall, MessageOp } from "./types.js";

/**
 * Get the message history relevant to a specific tool call.
 */
export function getHistoryForToolCall(
  messages: BaseMessage[],
  toolCallId: string
): BaseMessage[] {
  const results: BaseMessage[] = [];
  let seenAiMessage = false;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];

    if (msg instanceof AIMessage) {
      if (!seenAiMessage) {
        // Filter to only the relevant tool call
        const relevantCalls = (msg.tool_calls || []).filter(
          (tc) => tc.id === toolCallId
        );
        if (relevantCalls.length > 0) {
          const filtered = new AIMessage({
            content: String(msg.content),
            tool_calls: relevantCalls,
          });
          filtered.id = msg.id;
          results.unshift(filtered);
        }
      }
      seenAiMessage = true;
    } else if (msg instanceof ToolMessage) {
      if (msg.tool_call_id !== toolCallId && !seenAiMessage) {
        continue;
      }
      results.unshift(msg);
    } else {
      results.unshift(msg);
    }
  }

  return results;
}

/**
 * Apply message operations to update the message list.
 */
export function applyMessageOps(
  messages: BaseMessage[],
  ops: MessageOp[]
): BaseMessage[] {
  let result = [...messages];

  for (const op of ops) {
    switch (op.op) {
      case "delete": {
        const targetId = op.target as string;
        result = result.filter((m) => m.id !== targetId);
        break;
      }
      case "update_tool_call": {
        const target = op.target as ToolCall;
        result = result.map((m) => {
          if (m instanceof AIMessage) {
            const updatedCalls = (m.tool_calls || []).map((tc) =>
              tc.id === target.id ? target : tc
            );
            const updated = new AIMessage({
              content: m.content,
              tool_calls: updatedCalls as Array<{
                id: string;
                name: string;
                args: Record<string, unknown>;
              }>,
            });
            updated.id = m.id;
            return updated;
          }
          return m;
        });
        break;
      }
      case "update_tool_name": {
        const target = op.target as { id: string; name: string };
        result = result.map((m) => {
          if (m instanceof AIMessage) {
            const updatedCalls = (m.tool_calls || []).map((tc) =>
              tc.id === target.id ? { ...tc, name: target.name } : tc
            );
            const updated = new AIMessage({
              content: m.content,
              tool_calls: updatedCalls as Array<{
                id: string;
                name: string;
                args: Record<string, unknown>;
              }>,
            });
            updated.id = m.id;
            return updated;
          }
          return m;
        });
        break;
      }
    }
  }

  return result;
}
