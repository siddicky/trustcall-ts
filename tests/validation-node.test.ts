import { describe, it, expect } from "vitest";
import { z } from "zod";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { ValidationNode } from "../src/trustcall/validation-node.js";

describe("ValidationNode", () => {
  const UserSchema = z
    .object({
      name: z.string(),
      age: z.number().min(0).max(150),
    })
    .describe("User");

  const AddressSchema = z
    .object({
      street: z.string(),
      city: z.string(),
      zipCode: z.string().regex(/^\d{5}$/),
    })
    .describe("Address");

  describe("constructor", () => {
    it("should register Zod schemas by description", () => {
      const validator = new ValidationNode([UserSchema]);
      expect(validator.schemasByName.has("User")).toBe(true);
    });

    it("should register multiple schemas", () => {
      const validator = new ValidationNode([UserSchema, AddressSchema]);
      expect(validator.schemasByName.has("User")).toBe(true);
      expect(validator.schemasByName.has("Address")).toBe(true);
    });

    it("should use default name for schemas without description", () => {
      const UnnamedSchema = z.object({ foo: z.string() });
      const validator = new ValidationNode([UnnamedSchema]);
      expect(validator.schemasByName.has("Schema_0")).toBe(true);
    });

    it("should accept custom name option", () => {
      const validator = new ValidationNode([UserSchema], { name: "custom_validator" });
      expect(validator.name).toBe("custom_validator");
    });

    it("should accept tags option", () => {
      const validator = new ValidationNode([UserSchema], { tags: ["test", "validation"] });
      expect(validator.tags).toEqual(["test", "validation"]);
    });

    it("should use default name and empty tags", () => {
      const validator = new ValidationNode([UserSchema]);
      expect(validator.name).toBe("validation");
      expect(validator.tags).toEqual([]);
    });
  });

  describe("invoke with array input", () => {
    it("should validate valid tool calls", async () => {
      const validator = new ValidationNode([UserSchema]);
      const aiMessage = new AIMessage({
        content: "Here is the user info",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: 30 },
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(1);
      expect(result[0]).toBeInstanceOf(ToolMessage);
      expect(result[0].tool_call_id).toBe("call-1");
      expect(result[0].status).toBe("success");
      expect(result[0].additional_kwargs?.is_error).toBeUndefined();
    });

    it("should return error for invalid tool calls", async () => {
      const validator = new ValidationNode([UserSchema]);
      const aiMessage = new AIMessage({
        content: "Here is the user info",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: -5 }, // Invalid: negative age
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(result).toHaveLength(1);
      expect(result[0].status).toBe("error");
      expect(result[0].additional_kwargs?.is_error).toBe(true);
    });

    it("should return error for unrecognized tool name", async () => {
      const validator = new ValidationNode([UserSchema]);
      const aiMessage = new AIMessage({
        content: "Calling unknown tool",
        tool_calls: [
          {
            id: "call-1",
            name: "UnknownTool",
            args: { foo: "bar" },
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(result).toHaveLength(1);
      expect(result[0].status).toBe("error");
      expect(result[0].additional_kwargs?.is_error).toBe(true);
      expect(result[0].content).toContain("Unrecognized tool name");
      expect(result[0].content).toContain("UnknownTool");
      expect(result[0].content).toContain("User");
      expect(result[0].content).toContain("PatchFunctionName");
    });

    it("should validate multiple tool calls in parallel", async () => {
      const validator = new ValidationNode([UserSchema, AddressSchema]);
      const aiMessage = new AIMessage({
        content: "Multiple calls",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: 30 },
          },
          {
            id: "call-2",
            name: "Address",
            args: { street: "123 Main St", city: "NYC", zipCode: "12345" },
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(result).toHaveLength(2);
      expect(result[0].status).toBe("success");
      expect(result[1].status).toBe("success");
    });

    it("should return mix of valid and invalid results", async () => {
      const validator = new ValidationNode([UserSchema, AddressSchema]);
      const aiMessage = new AIMessage({
        content: "Mixed calls",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: 30 },
          },
          {
            id: "call-2",
            name: "Address",
            args: { street: "123 Main St", city: "NYC", zipCode: "invalid" },
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(result).toHaveLength(2);
      expect(result[0].status).toBe("success");
      expect(result[1].status).toBe("error");
      expect(result[1].additional_kwargs?.is_error).toBe(true);
    });
  });

  describe("invoke with object input", () => {
    it("should return object with messages property", async () => {
      const validator = new ValidationNode([UserSchema]);
      const aiMessage = new AIMessage({
        content: "Here is the user info",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: 30 },
          },
        ],
      });

      const result = await validator.invoke({ messages: [aiMessage] });
      expect(result).toHaveProperty("messages");
      expect(Array.isArray((result as { messages: ToolMessage[] }).messages)).toBe(true);
    });
  });

  describe("custom error formatter", () => {
    it("should use custom error formatter", async () => {
      const validator = new ValidationNode([UserSchema], {
        formatError: (error, call, _schema) =>
          `Custom error for ${call.name}: ${error.message}`,
      });

      const aiMessage = new AIMessage({
        content: "Invalid data",
        tool_calls: [
          {
            id: "call-1",
            name: "User",
            args: { name: "Alice", age: -5 },
          },
        ],
      });

      const result = await validator.invoke([aiMessage]);
      expect(result[0].content).toContain("Custom error for User:");
    });
  });

  describe("error handling", () => {
    it("should throw if last message is not AIMessage", async () => {
      const validator = new ValidationNode([UserSchema]);
      const humanMessage = new HumanMessage({ content: "Hello" });

      await expect(validator.invoke([humanMessage])).rejects.toThrow(
        "Last message is not an AIMessage"
      );
    });

    it("should handle AIMessage with no tool calls", async () => {
      const validator = new ValidationNode([UserSchema]);
      const aiMessage = new AIMessage({
        content: "Just a regular message",
      });

      const result = await validator.invoke([aiMessage]);
      expect(result).toHaveLength(0);
    });
  });
});
