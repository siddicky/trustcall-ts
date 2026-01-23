/**
 * Integration tests for TrustCallJS extractor with real LLM providers.
 *
 * These tests require actual API keys to be set in the environment:
 * - OPENAI_API_KEY for OpenAI tests
 * - ANTHROPIC_API_KEY for Anthropic tests
 *
 * Run with: pnpm test tests/integration
 *
 * To run only specific provider tests:
 * - OPENAI_API_KEY=xxx pnpm test tests/integration --grep "OpenAI"
 * - ANTHROPIC_API_KEY=xxx pnpm test tests/integration --grep "Anthropic"
 */

import { describe, it, expect, beforeAll } from "vitest";
import { z } from "zod";
import { createExtractor } from "../../src/trustcall/extractor.js";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

// Test schemas
const UserInfo = z
  .object({
    name: z.string().describe("User's full name"),
    age: z.number().min(0).max(150).describe("User's age in years"),
    email: z.string().email().optional().describe("User's email address"),
  })
  .describe("UserInfo");

const CompanyProfile = z
  .object({
    name: z.string().describe("Company name"),
    founded: z.number().min(1800).max(2030).describe("Year founded"),
    headquarters: z
      .object({
        city: z.string(),
        country: z.string(),
      })
      .describe("Company headquarters location"),
    products: z
      .array(z.string())
      .describe("Main products or services (at least one)"),
  })
  .describe("CompanyProfile");

const ContactInfo = z
  .object({
    firstName: z.string().describe("First name"),
    lastName: z.string().describe("Last name"),
    phone: z
      .string()
      .optional()
      .describe("Phone number in format like +1-555-123-4567"),
    address: z
      .object({
        street: z.string(),
        city: z.string(),
        state: z.string().optional(),
        country: z.string(),
        postalCode: z.string(),
      })
      .describe("Mailing address"),
  })
  .describe("ContactInfo");

const UserPreferences = z
  .object({
    userId: z.string().describe("User identifier"),
    theme: z.enum(["light", "dark", "system"]).describe("UI theme preference"),
    language: z.string().describe("Preferred language"),
    notifications: z
      .object({
        email: z.boolean(),
        push: z.boolean(),
        sms: z.boolean(),
      })
      .describe("Notification preferences"),
    favoriteColors: z.array(z.string()).describe("List of favorite colors"),
  })
  .describe("UserPreferences");

// Helper to check if API key is available
function hasApiKey(provider: "openai" | "anthropic"): boolean {
  const envVar =
    provider === "openai" ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY";
  return !!process.env[envVar];
}

// Helper to create LLM instance
async function createLlm(
  provider: "openai" | "anthropic",
  options?: { model?: string; temperature?: number }
): Promise<BaseChatModel> {
  const { model, temperature = 0 } = options || {};

  if (provider === "openai") {
    const { ChatOpenAI } = await import("@langchain/openai");
    return new ChatOpenAI({
      model: model || "gpt-4o-mini",
      temperature,
    });
  } else {
    const { ChatAnthropic } = await import("@langchain/anthropic");
    return new ChatAnthropic({
      model: model || "claude-3-haiku-20240307",
      temperature,
    });
  }
}

// Shared test suite that can be run against any provider
function runProviderTests(
  providerName: string,
  getLlm: () => Promise<BaseChatModel>
) {
  describe(`${providerName} Integration Tests`, () => {
    let llm: BaseChatModel;

    beforeAll(async () => {
      llm = await getLlm();
    });

    describe("Basic Extraction", () => {
      it("should extract simple user info from text", async () => {
        const extractor = createExtractor(llm, {
          tools: [UserInfo],
        });

        const result = await extractor.invoke({
          messages:
            "My name is Alice Johnson and I'm 28 years old. My email is alice@example.com",
        });

        expect(result.responses).toHaveLength(1);
        expect(result.responses[0]).toMatchObject({
          name: expect.stringContaining("Alice"),
          age: 28,
          email: "alice@example.com",
        });
        expect(result.attempts).toBeGreaterThanOrEqual(1);
      }, 30000);

      it("should extract user info without optional fields", async () => {
        // Use a schema without optional email to avoid null vs undefined issues
        const UserInfoRequired = z
          .object({
            name: z.string().describe("User's full name"),
            age: z.number().min(0).max(150).describe("User's age in years"),
          })
          .describe("UserInfoRequired");

        const extractor = createExtractor(llm, {
          tools: [UserInfoRequired],
        });

        const result = await extractor.invoke({
          messages: "Bob Smith is 45 years old. Extract his information.",
        });

        expect(result.responses).toHaveLength(1);
        expect(result.responses[0]).toMatchObject({
          name: expect.stringContaining("Bob"),
          age: 45,
        });
      }, 30000);

      it("should handle string input directly", async () => {
        const extractor = createExtractor(llm, {
          tools: [UserInfo],
        });

        const result = await extractor.invoke(
          "Charlie Brown, age 35, charlie@email.com"
        );

        expect(result.responses).toHaveLength(1);
        expect(result.responses[0].name).toContain("Charlie");
        expect(result.responses[0].age).toBe(35);
      }, 30000);
    });

    describe("Complex Schema Extraction", () => {
      it("should extract nested company profile", async () => {
        const extractor = createExtractor(llm, {
          tools: [CompanyProfile],
        });

        const result = await extractor.invoke({
          messages: `
            TechCorp was founded in 2015.
            Their headquarters is in San Francisco, USA.
            They make enterprise software and cloud services.
          `,
        });

        expect(result.responses).toHaveLength(1);
        const company = result.responses[0];
        expect(company.name).toBeDefined();
        expect(company.founded).toBe(2015);
        expect(company.headquarters).toMatchObject({
          city: expect.stringContaining("San Francisco"),
          country: expect.stringMatching(/USA|United States/i),
        });
        expect(company.products).toBeInstanceOf(Array);
        expect(company.products.length).toBeGreaterThan(0);
      }, 30000);

      it("should extract contact info with nested address", async () => {
        const extractor = createExtractor(llm, {
          tools: [ContactInfo],
        });

        const result = await extractor.invoke({
          messages: `
            Contact: John Doe
            Phone: +1-555-123-4567
            Address: 123 Main Street, New York, NY 10001, USA
          `,
        });

        expect(result.responses).toHaveLength(1);
        const contact = result.responses[0];
        expect(contact.firstName).toBe("John");
        expect(contact.lastName).toBe("Doe");
        expect(contact.phone).toBeDefined();
        expect(contact.address.city).toContain("New York");
        expect(contact.address.country).toMatch(/USA|United States/i);
      }, 30000);
    });

    describe("Multiple Schema Extraction", () => {
      it("should extract from multiple schemas in one call", async () => {
        const extractor = createExtractor(llm, {
          tools: [UserInfo, CompanyProfile],
        });

        const result = await extractor.invoke({
          messages: `
            Jane Smith (32 years old) works at DataCo.
            DataCo was founded in 2010 in Seattle, USA and makes analytics tools.
            Jane's email is jane@dataco.com
          `,
        });

        // Should extract at least one response
        expect(result.responses.length).toBeGreaterThanOrEqual(1);
      }, 30000);
    });

    describe("Updating Existing Data", () => {
      it("should update existing schema via patches", async () => {
        const extractor = createExtractor(llm, {
          tools: [UserPreferences],
          enableUpdates: true,
        });

        const existingData = {
          UserPreferences: {
            userId: "user-123",
            theme: "light" as const,
            language: "English",
            notifications: {
              email: true,
              push: true,
              sms: false,
            },
            favoriteColors: ["blue", "green"],
          },
        };

        const result = await extractor.invoke({
          messages:
            "Please change my theme to dark mode and add red to my favorite colors",
          existing: existingData,
        });

        expect(result.responses).toHaveLength(1);
        const updated = result.responses[0];
        expect(updated.theme).toBe("dark");
        expect(updated.favoriteColors).toContain("red");
        // Original data should be preserved
        expect(updated.userId).toBe("user-123");
        expect(updated.language).toBe("English");
        expect(updated.favoriteColors).toContain("blue");
        expect(updated.favoriteColors).toContain("green");
      }, 30000);

      it("should preserve unmentioned fields during update", async () => {
        const extractor = createExtractor(llm, {
          tools: [UserPreferences],
          enableUpdates: true,
        });

        const existingData = {
          UserPreferences: {
            userId: "user-456",
            theme: "system" as const,
            language: "Spanish",
            notifications: {
              email: false,
              push: true,
              sms: true,
            },
            favoriteColors: ["purple"],
          },
        };

        const result = await extractor.invoke({
          messages: "Turn off all my notifications",
          existing: existingData,
        });

        expect(result.responses).toHaveLength(1);
        const updated = result.responses[0];
        // Notifications should be updated
        expect(updated.notifications.email).toBe(false);
        expect(updated.notifications.push).toBe(false);
        expect(updated.notifications.sms).toBe(false);
        // Other fields should be preserved
        expect(updated.userId).toBe("user-456");
        expect(updated.theme).toBe("system");
        expect(updated.language).toBe("Spanish");
      }, 30000);
    });

    describe("Validation and Retry", () => {
      it("should retry and fix validation errors", async () => {
        // Schema with strict constraints
        const StrictUser = z
          .object({
            name: z.string().min(2).max(50),
            age: z.number().int().min(18).max(120),
            score: z.number().min(0).max(100),
          })
          .describe("StrictUser");

        const extractor = createExtractor(llm, {
          tools: [StrictUser],
        });

        const result = await extractor.invoke({
          messages:
            "User: Alice (25 years old) with a score of 85 out of 100",
        });

        expect(result.responses).toHaveLength(1);
        const user = result.responses[0];
        expect(user.name.length).toBeGreaterThanOrEqual(2);
        expect(user.age).toBeGreaterThanOrEqual(18);
        expect(user.age).toBeLessThanOrEqual(120);
        expect(user.score).toBeGreaterThanOrEqual(0);
        expect(user.score).toBeLessThanOrEqual(100);
      }, 30000);

      it("should handle complex validation constraints", async () => {
        const Event = z
          .object({
            title: z.string().describe("Event title"),
            startDate: z.string().describe("Start date in ISO format"),
            endDate: z.string().describe("End date in ISO format"),
            attendees: z
              .array(z.string())
              .describe("List of attendee email addresses"),
            maxCapacity: z.number().int().min(1).describe("Maximum attendees"),
          })
          .describe("Event");

        const extractor = createExtractor(llm, {
          tools: [Event],
        });

        const result = await extractor.invoke({
          messages: `
            Schedule a meeting called "Team Sync" on 2024-06-15.
            It ends the same day.
            Invite alice@company.com and bob@company.com.
            Maximum 10 people can attend.
          `,
        });

        expect(result.responses).toHaveLength(1);
        const event = result.responses[0];
        expect(event.title).toBeTruthy();
        expect(event.attendees).toContain("alice@company.com");
        expect(event.attendees).toContain("bob@company.com");
        expect(event.maxCapacity).toBe(10);
      }, 30000);
    });

    describe("Edge Cases", () => {
      it("should handle slightly ambiguous input gracefully", async () => {
        const SimpleUser = z
          .object({
            name: z.string().describe("User's name"),
            age: z.number().describe("User's age"),
          })
          .describe("SimpleUser");

        const extractor = createExtractor(llm, {
          tools: [SimpleUser],
        });

        const result = await extractor.invoke({
          messages: "Extract user info: Alex is approximately 30 years old.",
        });

        expect(result.responses).toHaveLength(1);
        expect(result.responses[0].name).toContain("Alex");
        expect(result.responses[0].age).toBeCloseTo(30, -1); // Within 10
      }, 30000);

      it("should extract from longer text", async () => {
        const extractor = createExtractor(llm, {
          tools: [CompanyProfile],
        });

        const result = await extractor.invoke({
          messages: `
            In a recent interview, the CEO discussed the company's history.
            "We started GreenTech back in 2008," she explained. "Our goal was
            to make renewable energy accessible to everyone." The company,
            headquartered in Berlin, Germany, has since expanded to offer
            solar panels, wind turbines, and energy storage solutions.
            "We're proud of how far we've come," she added.
          `,
        });

        expect(result.responses).toHaveLength(1);
        const company = result.responses[0];
        expect(company.name).toContain("GreenTech");
        expect(company.founded).toBe(2008);
        expect(company.headquarters.city).toContain("Berlin");
        expect(company.products.length).toBeGreaterThan(0);
      }, 30000);
    });

    describe("Configuration Options", () => {
      it("should respect max_attempts configuration", async () => {
        const SimpleUser = z
          .object({
            name: z.string().describe("User's name"),
            age: z.number().describe("User's age"),
          })
          .describe("SimpleUser");

        const extractor = createExtractor(llm, {
          tools: [SimpleUser],
        });

        const result = await extractor.invoke(
          { messages: "Extract: Test User is 25 years old." },
          { configurable: { max_attempts: 1 } }
        );

        expect(result.attempts).toBe(1);
      }, 30000);

      it("should work with toolChoice option", async () => {
        const PersonInfo = z
          .object({
            name: z.string().describe("Person's name"),
            age: z.number().describe("Person's age"),
          })
          .describe("PersonInfo");

        const OrgInfo = z
          .object({
            orgName: z.string().describe("Organization name"),
            yearFounded: z.number().describe("Year founded"),
          })
          .describe("OrgInfo");

        const extractor = createExtractor(llm, {
          tools: [PersonInfo, OrgInfo],
          toolChoice: "PersonInfo",
        });

        const result = await extractor.invoke({
          messages:
            "Extract person info: Alice is 30 years old and works at TechCorp founded in 2015.",
        });

        // Should only extract PersonInfo due to toolChoice
        expect(result.responses).toHaveLength(1);
        expect(result.responses[0]).toHaveProperty("name");
        expect(result.responses[0]).toHaveProperty("age");
      }, 30000);
    });
  });
}

// OpenAI Tests
const openaiAvailable = hasApiKey("openai");
describe.skipIf(!openaiAvailable)("OpenAI Provider", () => {
  runProviderTests("OpenAI", () => createLlm("openai"));

  // OpenAI-specific tests
  describe("OpenAI Specific", () => {
    const SimpleUser = z
      .object({
        name: z.string().describe("User's name"),
        age: z.number().describe("User's age"),
      })
      .describe("SimpleUser");

    it("should work with gpt-4o model", async () => {
      const llm = await createLlm("openai", { model: "gpt-4o" });
      const extractor = createExtractor(llm, {
        tools: [SimpleUser],
      });

      const result = await extractor.invoke({
        messages: "Extract user info: Maria Garcia is 42 years old.",
      });

      expect(result.responses).toHaveLength(1);
      expect(result.responses[0].name).toContain("Maria");
    }, 30000);

    it("should work with gpt-4o-mini model", async () => {
      const llm = await createLlm("openai", { model: "gpt-4o-mini" });
      const extractor = createExtractor(llm, {
        tools: [SimpleUser],
      });

      const result = await extractor.invoke({
        messages: "Extract user info: Tom Wilson is 38 years old.",
      });

      expect(result.responses).toHaveLength(1);
      expect(result.responses[0].name).toContain("Tom");
    }, 30000);
  });
});

// Anthropic Tests
const anthropicAvailable = hasApiKey("anthropic");
describe.skipIf(!anthropicAvailable)("Anthropic Provider", () => {
  runProviderTests("Anthropic", () => createLlm("anthropic"));

  // Anthropic-specific tests
  describe("Anthropic Specific", () => {
    const SimpleUser = z
      .object({
        name: z.string().describe("User's name"),
        age: z.number().describe("User's age"),
      })
      .describe("SimpleUser");

    it("should work with claude-3-haiku model", async () => {
      const llm = await createLlm("anthropic", {
        model: "claude-3-haiku-20240307",
      });
      const extractor = createExtractor(llm, {
        tools: [SimpleUser],
      });

      const result = await extractor.invoke({
        messages: "Extract user info: Sarah Chen is 29 years old.",
      });

      expect(result.responses).toHaveLength(1);
      expect(result.responses[0].name).toContain("Sarah");
    }, 30000);

    it("should work with claude-sonnet-4 model", async () => {
      const llm = await createLlm("anthropic", {
        model: "claude-sonnet-4-20250514",
      });
      const extractor = createExtractor(llm, {
        tools: [SimpleUser],
      });

      const result = await extractor.invoke({
        messages: "Extract user info: David Lee is 33 years old.",
      });

      expect(result.responses).toHaveLength(1);
      expect(result.responses[0].name).toContain("David");
    }, 30000);
  });
});

// Cross-provider comparison tests (only run if both providers are available)
const bothAvailable = openaiAvailable && anthropicAvailable;
describe.skipIf(!bothAvailable)("Cross-Provider Comparison", () => {
  it("should produce consistent results across providers", async () => {
    const testInput = "John Smith is 35 years old and his email is john@test.com";

    const openaiLlm = await createLlm("openai");
    const anthropicLlm = await createLlm("anthropic");

    const openaiExtractor = createExtractor(openaiLlm, { tools: [UserInfo] });
    const anthropicExtractor = createExtractor(anthropicLlm, {
      tools: [UserInfo],
    });

    const [openaiResult, anthropicResult] = await Promise.all([
      openaiExtractor.invoke({ messages: testInput }),
      anthropicExtractor.invoke({ messages: testInput }),
    ]);

    // Both should extract the same core data
    expect(openaiResult.responses[0].name).toContain("John");
    expect(anthropicResult.responses[0].name).toContain("John");
    expect(openaiResult.responses[0].age).toBe(35);
    expect(anthropicResult.responses[0].age).toBe(35);
  }, 60000);
});
