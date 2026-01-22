# 🤝 TrustCallJS

TypeScript port of [trustcall](https://github.com/hinthornw/trustcall) - Utilities for validated tool calling and extraction with retries using LLMs.

Built on top of [@langchain/langgraph](https://github.com/langchain-ai/langgraphjs).

## Installation

```bash
npm install trustcalljs @langchain/langgraph @langchain/core
```

## Why TrustCallJS?

[Tool calling](https://js.langchain.com/docs/how_to/tool_calling/) makes it easier to compose LLM calls within reliable software systems, but LLMs today can be error prone and inefficient in two common scenarios:

1. **Populating complex, nested schemas** - LLMs often make validation errors on deeply nested structures
2. **Updating existing schemas without information loss** - Regenerating entire objects can lose important data

TrustCallJS solves these problems using **JSONPatch** to correct validation errors, reducing costs and improving reliability.

## Quick Start

```typescript
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { createExtractor } from "trustcalljs";

// Define your schema
const UserInfo = z.object({
  name: z.string().describe("User's full name"),
  age: z.number().describe("User's age in years"),
}).describe("UserInfo");

// Create the extractor
const llm = new ChatOpenAI({ model: "gpt-4" });
const extractor = createExtractor(llm, {
  tools: [UserInfo],
});

// Extract structured data
const result = await extractor.invoke({
  messages: "My name is Alice and I'm 30 years old",
});

console.log(result.responses[0]);
// { name: "Alice", age: 30 }
```

## Features

### Complex Schema Extraction

TrustCallJS handles complex, deeply nested schemas that often cause validation errors:

```typescript
const TelegramPreferences = z.object({
  communication: z.object({
    telegram: z.object({
      preferredEncoding: z.enum(["morse", "standard"]),
      paperType: z.string().optional(),
    }),
    semaphore: z.object({
      flagColor: z.string(),
    }),
  }),
}).describe("TelegramPreferences");

const extractor = createExtractor(llm, {
  tools: [TelegramPreferences],
});

// Even with complex schemas, TrustCallJS will retry and fix validation errors
const result = await extractor.invoke({
  messages: `Extract preferences from: 
    User: I'd like morse code on daredevil paper`,
});
```

### Updating Existing Data

Update existing schemas without losing information:

```typescript
const UserPreferences = z.object({
  name: z.string(),
  favoriteColors: z.array(z.string()),
  settings: z.object({
    notifications: z.boolean(),
    theme: z.enum(["light", "dark"]),
  }),
}).describe("UserPreferences");

const existing = {
  UserPreferences: {
    name: "Alice",
    favoriteColors: ["blue", "green"],
    settings: {
      notifications: true,
      theme: "light",
    },
  },
};

const extractor = createExtractor(llm, {
  tools: [UserPreferences],
  enableUpdates: true,
});

const result = await extractor.invoke({
  messages: "I now prefer dark theme and add purple to my colors",
  existing,
});

// Result preserves existing data while applying updates:
// {
//   name: "Alice",
//   favoriteColors: ["blue", "green", "purple"],
//   settings: { notifications: true, theme: "dark" }
// }
```

### Validation and Retries

TrustCallJS automatically:
- Validates tool call outputs against your schemas
- Generates JSONPatch operations to fix validation errors
- Retries with corrections up to a configurable maximum

```typescript
const extractor = createExtractor(llm, {
  tools: [MySchema],
});

// Configure max retry attempts
const result = await extractor.invoke(
  { messages: "..." },
  { configurable: { max_attempts: 5 } }
);

// Check how many attempts were needed
console.log(`Extraction completed in ${result.attempts} attempts`);
```

## API Reference

### `createExtractor(llm, options)`

Creates an extractor runnable.

**Parameters:**
- `llm`: A LangChain chat model (e.g., `ChatOpenAI`, `ChatAnthropic`)
- `options`: Extractor configuration
  - `tools`: Array of Zod schemas, structured tools, or functions
  - `toolChoice?`: Force a specific tool to be used
  - `enableInserts?`: Allow creating new schemas when updating (default: false)
  - `enableUpdates?`: Allow updating existing schemas (default: true)
  - `enableDeletes?`: Allow deleting existing schemas (default: false)
  - `existingSchemaPolicy?`: How to handle unknown existing schemas (default: true)

**Returns:** An extractor with `invoke()` and `stream()` methods.

### `ExtractionOutputs`

```typescript
interface ExtractionOutputs {
  messages: AIMessage[];      // The AI messages with tool calls
  responses: unknown[];       // Validated schema instances
  responseMetadata: Array<{   // Metadata about each response
    id: string;
    jsonDocId?: string;
  }>;
  attempts: number;           // Number of extraction attempts
}
```

### `ValidationNode`

A standalone validation node for use in custom graphs:

```typescript
import { ValidationNode } from "trustcalljs";

const validator = new ValidationNode([UserInfo, Preferences], {
  formatError: (error, call, schema) => `Custom error: ${error.message}`,
});

const result = await validator.invoke({ messages });
```

## How It Works

1. **Initial Extraction**: The LLM generates tool calls based on input
2. **Validation**: Tool calls are validated against Zod schemas
3. **Error Detection**: Validation errors are detected and formatted
4. **Patch Generation**: The LLM generates JSONPatch operations to fix errors
5. **Application**: Patches are applied to the original arguments
6. **Retry**: The process repeats until validation passes or max attempts reached

This approach is more efficient than regenerating entire outputs because:
- Only the failing parts are regenerated
- Existing correct data is preserved
- Fewer output tokens are needed for fixes

## License

MIT