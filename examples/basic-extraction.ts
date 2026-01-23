/**
 * TrustcallTS - Basic Extraction Example
 *
 * Run with: deno run --allow-env --allow-net examples/basic-extraction.ts
 */

import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
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

  console.log("=== Example 1: Basic User Extraction ===\n");

  const userExtractor = createExtractor(llm, {
    tools: [UserInfo],
  });

  const userResult = await userExtractor.invoke({
    messages:
      "My name is Alice Johnson and I'm 30 years old. Email me at alice@example.com",
  });

  console.log("Extracted user:", JSON.stringify(userResult.responses[0], null, 2));
  console.log(`Attempts needed: ${userResult.attempts}\n`);

  console.log("=== Example 2: Complex Schema Extraction ===\n");

  const companyExtractor = createExtractor(llm, {
    tools: [CompanyProfile],
  });

  const companyResult = await companyExtractor.invoke({
    messages: `
      Anthropic is an AI safety company founded in 2021.
      They're headquartered in San Francisco, USA.
      Their main products are Claude (an AI assistant) and the Claude API.
    `,
  });

  console.log(
    "Extracted company:",
    JSON.stringify(companyResult.responses[0], null, 2)
  );
  console.log(`Attempts needed: ${companyResult.attempts}\n`);

  console.log("=== Example 3: Updating Existing Data ===\n");

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
    messages: "Switch to dark theme and add purple to my favorite colors",
    existing: existingData,
  });

  console.log(
    "\nUpdated data:",
    JSON.stringify(updateResult.responses[0], null, 2)
  );
  console.log(`Attempts needed: ${updateResult.attempts}`);
}

main().catch(console.error);
