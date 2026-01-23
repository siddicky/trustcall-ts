# Integration Tests

These integration tests verify TrustCallJS functionality with real LLM providers.

## Prerequisites

You need API keys for the providers you want to test:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable

## Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

The `.env` file is automatically loaded when running tests.

## Running Tests

### Run all integration tests

```bash
# With API keys in .env file
pnpm test:integration
```

### Run tests for a specific provider

```bash
# OpenAI only
OPENAI_API_KEY=your-key pnpm test:integration -- --grep "OpenAI"

# Anthropic only
ANTHROPIC_API_KEY=your-key pnpm test:integration -- --grep "Anthropic"
```

### Run a specific test

```bash
OPENAI_API_KEY=your-key pnpm test:integration -- --grep "should extract simple user info"
```

## Test Structure

The integration tests cover:

1. **Basic Extraction** - Simple schema extraction from text
2. **Complex Schema Extraction** - Nested objects and arrays
3. **Multiple Schema Extraction** - Extracting different schemas from one input
4. **Updating Existing Data** - Using JSONPatch to update existing schemas
5. **Validation and Retry** - Testing the retry mechanism for validation errors
6. **Edge Cases** - Ambiguous input, long text, etc.
7. **Configuration Options** - `max_attempts`, `toolChoice`, etc.
8. **Provider-Specific Tests** - Tests for specific models from each provider
9. **Cross-Provider Comparison** - Verifying consistent results across providers

## Test Behavior

- Tests are automatically skipped if the required API key is not set
- Each test has a 30-second timeout to accommodate LLM response times
- Cross-provider tests require both API keys and have a 60-second timeout

## Cost Considerations

These tests make real API calls, which incur costs. To minimize costs:

- Use `gpt-4o-mini` (OpenAI) and `claude-3-haiku` (Anthropic) by default
- Run specific tests during development instead of the full suite
- The test suite is designed to use minimal tokens while still being comprehensive
