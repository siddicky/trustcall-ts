/**
 * TrustCallJS - Basic Extraction Example
 *
 * Run with: deno run --allow-env --allow-net examples/basic-extraction.ts
 */

import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { createExtractor } from "trustcalljs";

// Define schemas for extraction
const UserInfo = z
  .object({
    name: z.string().describe("User's full name"),
    age: z.number().describe("User's age in years"),
    email: z.string().email().optional().describe("User's email address"),
  })
  .describe("UserInfo");

const CompanyProfile = z
  .object({
    name: z.string().describe("Company name"),
    founded: z.number().describe("Year founded"),
    headquarters: z
      .object({
        city: z.string(),
        country: z.string(),
      })
      .describe("Company headquarters"),
    products: z.array(z.string()).describe("Main products or services"),
  })
  .describe("CompanyProfile");

async function main() {
  // Check for API key
  const apiKey = Deno.env.get("OPENAI_API_KEY");
  if (!apiKey) {
    console.error("Please set OPENAI_API_KEY environment variable");
    Deno.exit(1);
  }

  // Initialize the LLM
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });

  // ==================================================
  // Example 1: String input (simplest)
  // ==================================================
  console.log("=== Example 1: String input ===\n");

  const userExtractor = createExtractor(llm, {
    tools: [UserInfo],
  });

  const userResult = await userExtractor.invoke(
    "My name is Alice Johnson and I'm 30 years old. Email me at alice@example.com",
  );

  console.log(
    "Extracted user:",
    JSON.stringify(userResult.responses[0], null, 2),
  );
  console.log(`Attempts needed: ${userResult.attempts}\n`);

  // ==================================================
  // Example 2: Single HumanMessage input
  // ==================================================
  console.log("=== Example 2: Single HumanMessage input ===\n");

  const companyExtractor = createExtractor(llm, {
    tools: [CompanyProfile],
  });

  const companyResult = await companyExtractor.invoke(
    new HumanMessage(
      `Anthropic is an AI safety company founded in 2021.
They're headquartered in San Francisco, USA.
Their main products are Claude (an AI assistant) and the Claude API.`,
    ),
  );

  console.log(
    "Extracted company:",
    JSON.stringify(companyResult.responses[0], null, 2),
  );
  console.log(`Attempts needed: ${companyResult.attempts}\n`);

  // ==================================================
  // Example 3: Array of BaseMessage (LangGraph MessagesValue)
  // ==================================================
  console.log("=== Example 3: Array of BaseMessage (LangGraph style) ===\n");

  const msgArrayResult = await companyExtractor.invoke({
    messages: [
      new HumanMessage(
        "Google was founded in 1998 in Mountain View, USA. They make Search, Chrome, and Android.",
      ),
    ],
  });

  console.log(
    "Extracted company:",
    JSON.stringify(msgArrayResult.responses[0], null, 2),
  );
  console.log(`Attempts needed: ${msgArrayResult.attempts}\n`);

  // ==================================================
  // Example 4: OpenAI-style message dict format
  // ==================================================
  console.log("=== Example 4: OpenAI-style message dict format ===\n");

  const dictResult = await companyExtractor.invoke({
    messages: [
      {
        role: "user",
        content:
          "Microsoft was founded in 1975 in Redmond, USA. Products include Windows, Office, and Azure.",
      },
    ],
  });

  console.log(
    "Extracted company:",
    JSON.stringify(dictResult.responses[0], null, 2),
  );
  console.log(`Attempts needed: ${dictResult.attempts}\n`);

  // ==================================================
  // Example 5: Updating existing data with patches
  // ==================================================
  console.log("=== Example 5: Updating Existing Data ===\n");

  const UserPreferences = z
    .object({
      name: z.string(),
      favoriteColors: z.array(z.string()),
      settings: z.object({
        theme: z.enum(["light", "dark", "system"]),
        notifications: z.boolean(),
      }),
    })
    .describe("UserPreferences");

  const existingData = {
    UserPreferences: {
      name: "Alice",
      favoriteColors: ["blue", "green"],
      settings: {
        theme: "light" as const,
        notifications: true,
      },
    },
  };

  console.log("Existing data:", JSON.stringify(existingData, null, 2));

  const updateExtractor = createExtractor(llm, {
    tools: [UserPreferences],
    enableUpdates: true,
  });

  const updateResult = await updateExtractor.invoke({
    messages: [
      {
        role: "user",
        content: "Switch to dark theme and add purple to my favorite colors",
      },
    ],
    existing: existingData,
  });

  console.log(
    "\nUpdated data:",
    JSON.stringify(updateResult.responses[0], null, 2),
  );
  console.log(`Attempts needed: ${updateResult.attempts}`);
}

main().catch(console.error);
