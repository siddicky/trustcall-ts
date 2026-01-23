# TrustCallJS Examples

Interactive Jupyter notebooks and TypeScript examples demonstrating TrustCallJS features.

## Prerequisites

1. **Install Deno** (v1.37+):
   ```bash
   curl -fsSL https://deno.land/install.sh | sh
   ```

2. **Build TrustCallJS** (from project root):
   ```bash
   pnpm install
   pnpm build
   ```

3. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # or for Anthropic
   export ANTHROPIC_API_KEY="your-api-key"
   ```

## Running TypeScript Examples

```bash
# Run the basic extraction example
deno run --allow-env --allow-net examples/basic-extraction.ts
```

## Running the Notebooks

### Install Jupyter (optional, for notebooks)

```bash
pip install jupyter
# or
brew install jupyter

# Install Deno Jupyter kernel
deno jupyter --install
```

### VS Code (Recommended)

1. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
2. Open any `.ipynb` file
3. Select "Deno" as the kernel (top-right dropdown)
4. Run cells with `Shift+Enter`

### JetBrains IDEs

Jupyter notebooks work out of the box in PyCharm, IntelliJ, and WebStorm with the Jupyter plugin.

### Command Line

```bash
# Start Jupyter Lab
jupyter lab

# Or classic notebook
jupyter notebook
```

Then navigate to the `examples/` directory and open any notebook.

## Available Examples

| File | Description |
|------|-------------|
| `basic-extraction.ts` | Core extraction examples with all input formats |
| `getting-started.ipynb` | Interactive introduction to TrustCallJS basics |

## Input Formats

The examples demonstrate all supported input formats:

```typescript
// 1. Simple string
const result = await extractor.invoke("My name is Alice");

// 2. Single BaseMessage
const result = await extractor.invoke(new HumanMessage("My name is Alice"));

// 3. Array of BaseMessage (LangGraph MessagesValue compatible)
const result = await extractor.invoke({
  messages: [new HumanMessage("My name is Alice")],
});

// 4. OpenAI-style message dict format
const result = await extractor.invoke({
  messages: [{ role: "user", content: "My name is Alice" }],
});
```

## Notes

- All code in Deno Jupyter runs with `--allow-all` permissions
- TypeScript is fully supported in the Deno kernel
- You can import npm packages using `npm:` specifier
