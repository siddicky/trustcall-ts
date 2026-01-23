# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TrustCallJS is a TypeScript port of the Python [trustcall](https://github.com/hinthornw/trustcall) library. It provides utilities for validated tool calling and extraction with retries using LLMs, built on top of @langchain/langgraph.

The library solves two common LLM challenges:
1. **Complex nested schema extraction** - LLMs often make validation errors on deeply nested structures
2. **Schema updates without data loss** - Regenerating entire objects can lose important data

The solution uses **JSONPatch** (RFC 6902) to correct validation errors, reducing costs and improving reliability.

## Build and Development Commands

```bash
pnpm install          # Install dependencies
pnpm build            # Compile TypeScript (tsc)
pnpm test             # Run tests (vitest)
pnpm test --run       # Run tests once (no watch)
pnpm lint             # Run ESLint on src/
```

## Examples (Deno Jupyter Notebooks)

Interactive examples are in `examples/` directory using Deno Jupyter notebooks.

```bash
# Install Deno Jupyter kernel (one-time setup)
deno jupyter --install

# Open notebooks in VS Code with Jupyter extension, or run:
jupyter lab
```

See `examples/README.md` for detailed setup instructions.

## Architecture

### Core Components

**`src/trustcall/extractor.ts`** - Main entry point via `createExtractor()`. Builds a LangGraph StateGraph with nodes:
- `extract` - Initial extraction without existing data
- `extractUpdates` - Update existing schemas using PatchDoc
- `validate` - Validate tool calls against Zod schemas
- `patch` - Fix validation errors using PatchFunctionErrors

The graph flow: START → extract/extractUpdates → validate → (patch if errors) → END

**`src/trustcall/validation-node.ts`** - `ValidationNode` class that validates AIMessage tool calls against Zod schemas. Returns ToolMessages with validation results or formatted errors.

**`src/trustcall/json-patch.ts`** - RFC 6902 JSONPatch implementation. `applyJsonPatches()` handles add, remove, replace, move, copy, test operations. `ensurePatches()` normalizes patch input from LLM responses.

**`src/trustcall/schemas.ts`** - Zod schemas for the patch system:
- `PatchDocSchema` - Updates to existing documents
- `PatchFunctionErrorsSchema` - Fixes for validation errors
- `createPatchFunctionNameSchema()` - Fixes for incorrect tool names
- `createRemoveDocSchema()` - Document deletion with allowed IDs

**`src/trustcall/types.ts`** - Core type definitions including `SchemaInstance`, `ExistingType`, `ToolType`, `JsonPatchOp`, `ToolCall`.

### Extraction Flow

1. User provides input messages and optional existing schemas
2. LLM generates tool calls based on provided Zod schemas
3. ValidationNode validates against schemas
4. If errors, LLM generates JSONPatch operations to fix them
5. Patches are applied and revalidated
6. Process repeats until valid or max_attempts reached (default: 3)

### Input Formats

The `createExtractor()` function accepts these input formats via `invoke()` and `stream()`:

1. **String** - Converted to `HumanMessage` internally
   ```typescript
   extractor.invoke("My name is Alice")
   ```

2. **Single BaseMessage** - Any LangChain message type
   ```typescript
   extractor.invoke(new HumanMessage("My name is Alice"))
   ```

3. **Object with messages array** - For LangGraph `MessagesValue` compatibility
   ```typescript
   // Array of BaseMessage instances
   extractor.invoke({ messages: [new HumanMessage("...")] })

   // OpenAI-style message dicts
   extractor.invoke({ messages: [{ role: "user", content: "..." }] })
   ```

4. **With existing data** - For schema updates
   ```typescript
   extractor.invoke({
     messages: [{ role: "user", content: "..." }],
     existing: { SchemaName: { ... } }
   })
   ```

### Key Patterns

- Uses `zod-to-json-schema` to convert Zod schemas to OpenAI function format
- State managed via LangGraph's `Annotation.Root` with custom reducers
- Tool calls identified by `id` field, tracked across patch/validation cycles
- Existing data can be dict-style `{SchemaName: {...}}` or array of `SchemaInstance`
- Uses LangChain's `isAIMessage()` and `isBaseMessage()` for duck-typed message detection (avoids `instanceof` issues across package boundaries)
