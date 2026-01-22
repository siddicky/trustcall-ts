# 🤝 trustcall (TypeScript)

LLMs struggle when asked to generate or modify large JSON blobs. **`trustcall` (TS)** solves this by asking the LLM to generate **[JSON Patch](https://datatracker.ietf.org/doc/html/rfc6902)** operations instead of full objects.

Generating patches is a simpler task that can be done iteratively, enabling:

- ⚡ Faster & cheaper generation of structured output
- 🐺 Resilient retrying of validation errors, even for **complex, nested schemas**
- 🧩 Accurate updates to **existing JSON documents**, avoiding undesired deletions

It works flexibly across common LLM workflows:

- ✂️ Extraction
- 🧭 LLM routing
- 🤖 Multi-step agent tool use

---

## Installation

```bash
npm install trustcall
# or
pnpm add trustcall
# or
yarn add trustcall
```

### Peer deps (recommended)

`trustcall` is framework-friendly. Choose what you use:

- **LangChainJS** tool-calling: `@langchain/core` + your model provider
- **Zod** for schema validation: `zod`
- JSON Patch applier: `fast-json-patch` (optional, if you want to customize patch application)

Example:

```bash
pnpm add @langchain/core zod fast-json-patch
```

---

## Usage

- [Extracting complex schemas](#complex-schema)
- [Updating schemas](#updating-schemas)
- [Simultaneous updates & insertions](#simultaneous-generation--updating)

### Quickstart

```ts
import { createExtractor } from "trustcall";
import { z } from "zod";

// 1) Define a schema (Zod recommended)
const Preferences = z.object({
  foods: z.array(z.string()).min(3, "Must have at least three favorite foods"),
});

type Preferences = z.infer<typeof Preferences>;

// 2) Create an extractor for a tool/schema
const extractor = createExtractor({
  llm, // any tool-calling LLM client supported by your runtime
  tools: [{ name: "Preferences", schema: Preferences }],
  toolChoice: "Preferences",
});

// 3) Invoke
const result = await extractor.invoke({
  messages: [{ role: "user", content: "I like apple pie and ice cream." }],
});

console.log(result.responses[0]);
// -> { foods: ["apple pie", "ice cream", "pizza", "sushi"] }
```

---

## Why trustcall?

Tool calling makes it easier to compose LLM calls within reliable software systems, but today’s LLMs can be error-prone and inefficient in two scenarios:

1. **Populating complex, nested schemas**
2. **Updating existing JSON objects without information loss**

These problems get worse when you want to handle **multiple tool calls**.

**`trustcall` increases structured extraction reliability without restricting you to a subset of JSON schema.**

The key idea:

> **Patch, don’t post.**  
> When output is invalid, ask the model to generate a *small JSON Patch* that fixes the error, instead of regenerating the entire document.

---

## Complex schema

Take a nested schema example (TypeScript + Zod):

<details>
<summary>Schema definition</summary>

```ts
import { z } from "zod";

const OutputFormat = z.object({
  preference: z.string(),
  sentencePreferenceRevealed: z.string(),
});

const TelegramPreferences = z.object({
  preferredEncoding: z.array(OutputFormat).nullable().optional(),
  favoriteTelegramOperators: z.array(OutputFormat).nullable().optional(),
  preferredTelegramPaper: z.array(OutputFormat).nullable().optional(),
});

const MorseCode = z.object({
  preferredKeyType: z.array(OutputFormat).nullable().optional(),
  favoriteMorseAbbreviations: z.array(OutputFormat).nullable().optional(),
});

const Semaphore = z.object({
  preferredFlagColor: z.array(OutputFormat).nullable().optional(),
  semaphoreSkillLevel: z.array(OutputFormat).nullable().optional(),
});

const TrustFallPreferences = z.object({
  preferredFallHeight: z.array(OutputFormat).nullable().optional(),
  trustLevel: z.array(OutputFormat).nullable().optional(),
  preferredCatchingTechnique: z.array(OutputFormat).nullable().optional(),
});

const CommunicationPreferences = z.object({
  telegram: TelegramPreferences,
  morseCode: MorseCode,
  semaphore: Semaphore,
});

const UserPreferences = z.object({
  communicationPreferences: CommunicationPreferences,
  trustFallPreferences: TrustFallPreferences,
});

export const TelegramAndTrustFallPreferences = z.object({
  pertinentUserPreferences: UserPreferences,
});
```

</details>

Naively tool-calling a schema like this often fails due to missing nested objects, `null` vs `{}`, or partial updates.

With `trustcall`, the model can converge reliably by iterating on **patches**.

```ts
import { createExtractor } from "trustcall";
import { TelegramAndTrustFallPreferences } from "./schema";

const extractor = createExtractor({
  llm,
  tools: [{ name: "TelegramAndTrustFallPreferences", schema: TelegramAndTrustFallPreferences }],
  toolChoice: "TelegramAndTrustFallPreferences",
});

const conversation = `Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir.`;

const result = await extractor.invoke({
  messages: [
    {
      role: "user",
      content: `Extract the preferences from the following conversation:
<convo>
${conversation}
</convo>`,
    },
  ],
});

console.log(result.responses[0]);
```

Example output:

```json
{
  "pertinentUserPreferences": {
    "communicationPreferences": {
      "telegram": {
        "preferredEncoding": [
          { "preference": "morse", "sentencePreferenceRevealed": "Morse, please." }
        ],
        "preferredTelegramPaper": [
          {
            "preference": "Daredevil",
            "sentencePreferenceRevealed": "Shall I use our \"Daredevil\" paper for this daring message?"
          }
        ]
      },
      "morseCode": {
        "preferredKeyType": [
          { "preference": "straight key", "sentencePreferenceRevealed": "I love using a straight key." }
        ]
      },
      "semaphore": {
        "preferredFlagColor": null,
        "semaphoreSkillLevel": null
      }
    },
    "trustFallPreferences": {
      "preferredFallHeight": [
        { "preference": "higher", "sentencePreferenceRevealed": "I'm ready for a higher fall." }
      ],
      "preferredCatchingTechnique": [
        {
          "preference": "diamond formation",
          "sentencePreferenceRevealed": "I prefer the diamond formation for catching."
        }
      ]
    }
  }
}
```

What’s different?

- `trustcall` validates tool output
- If invalid, it prompts the model for a **JSON Patch**
- Applies patch → re-validates → repeats until valid (or max retries)

This is typically **more reliable** and **cheaper** than regenerating the entire object repeatedly.

---

## Updating schemas

Many tasks require modifying an existing JSON doc based on new information (e.g., “memory updates” / “profile enrichment”).

The naive approach—“rewrite the entire object”—frequently drops fields due to omission or partial recall.

`trustcall` lets the LLM **focus only on what changed**.

### Example: User profile updates

```ts
import { z } from "zod";
import { createExtractor } from "trustcall";

const Address = z.object({
  street: z.string(),
  city: z.string(),
  country: z.string(),
  postalCode: z.string(),
});

const Pet = z.object({
  kind: z.string(),
  name: z.string().nullable().optional(),
  age: z.number().int().nullable().optional(),
});

const Hobby = z.object({
  name: z.string(),
  skillLevel: z.string(),
  frequency: z.string(),
});

const FavoriteMedia = z.object({
  shows: z.array(z.string()),
  movies: z.array(z.string()),
  books: z.array(z.string()),
});

const User = z.object({
  preferredName: z.string(),
  favoriteMedia: FavoriteMedia,
  favoriteFoods: z.array(z.string()),
  hobbies: z.array(Hobby),
  age: z.number().int(),
  occupation: z.string(),
  address: Address,
  favoriteColor: z.string().nullable().optional(),
  pets: z.array(Pet).nullable().optional(),
  languages: z.record(z.string()).default({}),
});

const initialUser = {
  preferredName: "Alex",
  favoriteMedia: {
    shows: ["Friends", "Game of Thrones", "Breaking Bad", "The Office", "Stranger Things"],
    movies: ["The Shawshank Redemption", "Inception", "The Dark Knight"],
    books: ["1984", "To Kill a Mockingbird", "The Great Gatsby"],
  },
  favoriteFoods: ["sushi", "pizza", "tacos", "ice cream", "pasta", "curry"],
  hobbies: [
    { name: "reading", skillLevel: "expert", frequency: "daily" },
    { name: "hiking", skillLevel: "intermediate", frequency: "weekly" },
  ],
  age: 28,
  occupation: "Software Engineer",
  address: { street: "123 Tech Lane", city: "San Francisco", country: "USA", postalCode: "94105" },
  favoriteColor: "blue",
  pets: [{ kind: "cat", name: "Luna", age: 3 }],
  languages: { English: "native", Spanish: "intermediate", Python: "expert" },
};

const extractor = createExtractor({
  llm,
  tools: [{ name: "User", schema: User }],
});

const conversation = `... new conversation text ...`;

const updated = await extractor.invoke({
  messages: [
    {
      role: "user",
      content: `Update the memory (JSON doc) to incorporate new information from the following conversation:
<convo>
${conversation}
</convo>`,
    },
  ],
  existing: { User: initialUser },
});

console.log(updated.responses[0]);
```

Under the hood, the model produces a JSON Patch against the existing `User` document rather than rewriting everything.

---

## Simultaneous generation & updating

Both problems above are compounded when you want the LLM to do:

- updates to existing docs **and**
- inserts for **new docs**

This is common when extracting “events” or “entities” from a conversation.

### Example: Updating & inserting people records

```ts
import { z } from "zod";
import { createExtractor } from "trustcall";

const Person = z.object({
  name: z.string(),
  relationship: z.string(),
  notes: z.array(z.string()),
});

const initialPeople = [
  { name: "Emma Thompson", relationship: "College friend", notes: ["Loves hiking", "Works in marketing"] },
  { name: "Michael Chen", relationship: "Coworker", notes: ["Vegetarian", "Plays guitar"] },
  { name: "Sarah Johnson", relationship: "Neighbor", notes: ["Loves gardening", "Makes amazing cookies"] },
];

// trustcall expects tuples for multiple existing docs:
// [jsonDocId, toolName, existingJson]
const existing = initialPeople.map((p, i) => [String(i), "Person", p] as const);

const extractor = createExtractor({
  llm,
  tools: [{ name: "Person", schema: Person }],
  toolChoice: "any",
  enableInserts: true,
});

const conversation = `... conversation text ...`;

const result = await extractor.invoke({
  messages: [
    {
      role: "user",
      content: `Update existing person records and create new ones based on the following conversation:\n\n${conversation}`,
    },
  ],
  existing,
});

for (let i = 0; i < result.responses.length; i++) {
  const meta = result.responseMetadata[i];
  console.log("ID:", meta.jsonDocId ?? "New");
  console.log(result.responses[i]);
}
```

You’ll get:

- updated docs for IDs `0`, `1`, `2`
- plus a **new inserted** `Person` for anyone newly mentioned

---

## API Overview

### `createExtractor(options)`

```ts
import type { JSONSchema7 } from "json-schema"; // optional
import type { ZodSchema } from "zod";           // optional

type TrustcallTool =
  | { name: string; schema: ZodSchema<any> }
  | { name: string; schema: JSONSchema7 }
  | { name: string; handler: (...args: any[]) => any }; // function-style tools

createExtractor({
  llm,
  tools: TrustcallTool[],
  toolChoice?: string | "any",
  enableInserts?: boolean,
  maxRetries?: number,
  patchStrategy?: "rfc6902",
  onRetry?: (info) => void,
});
```

### `invoke(payload)`

```ts
extractor.invoke({
  messages: Array<{ role: "user" | "assistant" | "system" | "tool"; content: string }>,
  existing?: Record<string, unknown> | Array<[string, string, unknown]>,
});
```

Returns:

- `messages`: LLM messages including tool-call payloads
- `responses`: validated tool outputs (typed objects)
- `responseMetadata`: per-response metadata including doc IDs / patch history (implementation-specific)

---

## How it works (high-level)

`trustcall` implements the “patch loop”:

1. Prompt the LLM to generate tool calls (structured args)
2. Validate tool call args against your schema(s)
3. If invalid:
   - ask the LLM to generate a **JSON Patch** that fixes the specific error(s)
   - apply patch to the last known object
   - validate again
4. Return only validated tool outputs

When `existing` is provided:

1. Prompt the LLM to generate JSON Patches against existing docs
2. Apply patches
3. Validate
4. Retry with more patches as needed

---

## More Examples

`trustcall` is designed to integrate cleanly with tool-calling runtimes:

- LangChainJS agents
- OpenAI Responses API / tool calls
- Any system where you can:
  1) request tool-call JSON  
  2) validate it  
  3) ask for patches on error

If you’re building conversational agents, the returned messages + validated tool outputs make it straightforward to plug into state machines, routers, or multi-tool workflows.

---

## Evaluating

A small evaluation benchmark is provided in:

- `./tests/evals/`

To run:

```bash
pnpm test
pnpm evals
```

(Exact commands depend on your repo setup and provider keys.)

---

## Contributing

PRs welcome.

Recommended dev setup:

```bash
pnpm i
pnpm lint
pnpm test
```

Guidelines:

- Keep the public API small
- Maintain deterministic patch application
- Prefer adapter-style schema support (Zod/JSON Schema/etc.)

---

## License

MIT
