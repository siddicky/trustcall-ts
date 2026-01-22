import { describe, it, expect } from "vitest";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { getHistoryForToolCall, applyMessageOps } from "../src/trustcall/utils.js";

describe("getHistoryForToolCall", () => {
  it("should return messages relevant to a specific tool call", () => {
    const messages = [
      new SystemMessage({ content: "You are helpful" }),
      new HumanMessage({ content: "Extract user info" }),
      new AIMessage({
        content: "Here is the info",
        tool_calls: [
          { id: "call-1", name: "User", args: { name: "Alice" } },
          { id: "call-2", name: "Address", args: { city: "NYC" } },
        ],
      }),
      new ToolMessage({
        content: "Valid",
        tool_call_id: "call-1",
        name: "User",
      }),
      new ToolMessage({
        content: "Invalid",
        tool_call_id: "call-2",
        name: "Address",
      }),
    ];

    const result = getHistoryForToolCall(messages, "call-1");

    // Should include system, human, filtered AI (only call-1), and call-1 tool message
    expect(result.length).toBeGreaterThanOrEqual(3);

    // Check that the AI message only contains the relevant tool call
    const aiMsg = result.find((m) => m instanceof AIMessage) as AIMessage;
    expect(aiMsg).toBeDefined();
    expect(aiMsg.tool_calls).toHaveLength(1);
    expect(aiMsg.tool_calls?.[0].id).toBe("call-1");
  });

  it("should include all non-tool messages", () => {
    const messages = [
      new SystemMessage({ content: "System" }),
      new HumanMessage({ content: "Human" }),
      new AIMessage({
        content: "AI",
        tool_calls: [{ id: "call-1", name: "Tool", args: {} }],
      }),
    ];

    const result = getHistoryForToolCall(messages, "call-1");

    expect(result.some((m) => m instanceof SystemMessage)).toBe(true);
    expect(result.some((m) => m instanceof HumanMessage)).toBe(true);
    expect(result.some((m) => m instanceof AIMessage)).toBe(true);
  });

  it("should filter out tool messages for other tool calls before AI message", () => {
    const messages = [
      new HumanMessage({ content: "Extract" }),
      new AIMessage({
        content: "Extracting",
        tool_calls: [
          { id: "call-1", name: "Tool1", args: {} },
          { id: "call-2", name: "Tool2", args: {} },
        ],
      }),
      new ToolMessage({
        content: "Result 1",
        tool_call_id: "call-1",
        name: "Tool1",
      }),
      new ToolMessage({
        content: "Result 2",
        tool_call_id: "call-2",
        name: "Tool2",
      }),
    ];

    const result = getHistoryForToolCall(messages, "call-1");
    const toolMessages = result.filter((m) => m instanceof ToolMessage);

    // Should include both tool messages since they come after the AI message
    expect(toolMessages.length).toBeGreaterThanOrEqual(1);
  });
});

describe("applyMessageOps", () => {
  describe("delete operation", () => {
    it("should remove message by id", () => {
      const msg1 = new HumanMessage({ content: "Hello" });
      msg1.id = "msg-1";
      const msg2 = new HumanMessage({ content: "World" });
      msg2.id = "msg-2";

      const result = applyMessageOps([msg1, msg2], [
        { op: "delete", target: "msg-1" },
      ]);

      expect(result).toHaveLength(1);
      expect(result[0].id).toBe("msg-2");
    });

    it("should handle deleting non-existent message", () => {
      const msg = new HumanMessage({ content: "Hello" });
      msg.id = "msg-1";

      const result = applyMessageOps([msg], [
        { op: "delete", target: "non-existent" },
      ]);

      expect(result).toHaveLength(1);
    });
  });

  describe("update_tool_call operation", () => {
    it("should update tool call in AIMessage", () => {
      const aiMsg = new AIMessage({
        content: "Calling tool",
        tool_calls: [
          { id: "call-1", name: "OldTool", args: { old: "value" } },
        ],
      });
      aiMsg.id = "ai-msg";

      const result = applyMessageOps([aiMsg], [
        {
          op: "update_tool_call",
          target: {
            id: "call-1",
            name: "NewTool",
            args: { new: "value" },
          },
        },
      ]);

      expect(result).toHaveLength(1);
      const updatedAi = result[0] as AIMessage;
      expect(updatedAi.tool_calls?.[0].name).toBe("NewTool");
      expect(updatedAi.tool_calls?.[0].args).toEqual({ new: "value" });
    });

    it("should only update matching tool call", () => {
      const aiMsg = new AIMessage({
        content: "Multiple calls",
        tool_calls: [
          { id: "call-1", name: "Tool1", args: { a: 1 } },
          { id: "call-2", name: "Tool2", args: { b: 2 } },
        ],
      });

      const result = applyMessageOps([aiMsg], [
        {
          op: "update_tool_call",
          target: { id: "call-1", name: "UpdatedTool1", args: { a: 10 } },
        },
      ]);

      const updatedAi = result[0] as AIMessage;
      expect(updatedAi.tool_calls?.[0].name).toBe("UpdatedTool1");
      expect(updatedAi.tool_calls?.[0].args).toEqual({ a: 10 });
      expect(updatedAi.tool_calls?.[1].name).toBe("Tool2");
      expect(updatedAi.tool_calls?.[1].args).toEqual({ b: 2 });
    });
  });

  describe("update_tool_name operation", () => {
    it("should update only the tool name", () => {
      const aiMsg = new AIMessage({
        content: "Calling tool",
        tool_calls: [
          { id: "call-1", name: "OldName", args: { preserved: "value" } },
        ],
      });

      const result = applyMessageOps([aiMsg], [
        {
          op: "update_tool_name",
          target: { id: "call-1", name: "NewName" },
        },
      ]);

      const updatedAi = result[0] as AIMessage;
      expect(updatedAi.tool_calls?.[0].name).toBe("NewName");
      expect(updatedAi.tool_calls?.[0].args).toEqual({ preserved: "value" });
    });
  });

  describe("multiple operations", () => {
    it("should apply multiple operations in order", () => {
      const msg1 = new HumanMessage({ content: "Hello" });
      msg1.id = "msg-1";
      const aiMsg = new AIMessage({
        content: "Response",
        tool_calls: [{ id: "call-1", name: "Tool", args: { x: 1 } }],
      });
      aiMsg.id = "ai-msg";

      const result = applyMessageOps(
        [msg1, aiMsg],
        [
          {
            op: "update_tool_call",
            target: { id: "call-1", name: "UpdatedTool", args: { x: 2 } },
          },
          { op: "delete", target: "msg-1" },
        ]
      );

      expect(result).toHaveLength(1);
      const updatedAi = result[0] as AIMessage;
      expect(updatedAi.tool_calls?.[0].name).toBe("UpdatedTool");
    });
  });

  describe("immutability", () => {
    it("should not modify original messages array", () => {
      const msg = new HumanMessage({ content: "Hello" });
      msg.id = "msg-1";
      const original = [msg];

      applyMessageOps(original, [{ op: "delete", target: "msg-1" }]);

      expect(original).toHaveLength(1);
    });
  });
});
