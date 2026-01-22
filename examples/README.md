# TrustCallJS Examples

Interactive Jupyter notebooks demonstrating TrustCallJS features using Deno.

## Prerequisites

1. **Install Deno** (v1.37+):
   ```bash
   curl -fsSL https://deno.land/install.sh | sh
   ```

2. **Install Jupyter**:
   ```bash
   pip install jupyter
   # or
   brew install jupyter
   ```

3. **Install Deno Jupyter kernel**:
   ```bash
   deno jupyter --install
   ```

4. **Build TrustCallJS** (from project root):
   ```bash
   pnpm install
   pnpm build
   ```

5. **Set up API keys**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   # or for Anthropic
   export ANTHROPIC_API_KEY="your-api-key"
   ```

## Running the Notebooks

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

| Notebook | Description |
|----------|-------------|
| `getting-started.ipynb` | Introduction to TrustCallJS basics |

## Notes

- All code in Deno Jupyter runs with `--allow-all` permissions
- TypeScript is fully supported in the Deno kernel
- You can import npm packages using `npm:` specifier
