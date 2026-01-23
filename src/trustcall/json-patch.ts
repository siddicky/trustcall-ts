import type { JsonPatchOp } from "./types.js";

/**
 * Resolve a JSON Pointer path to get the value at that location.
 */
function resolvePointer(doc: Record<string, unknown>, path: string): unknown {
  const parts = path.split("/").filter((p) => p !== "");
  let current: unknown = doc;

  for (const part of parts) {
    if (current === null || current === undefined) {
      return undefined;
    }
    if (Array.isArray(current)) {
      const index = parseInt(part, 10);
      current = current[index];
    } else if (typeof current === "object") {
      current = (current as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }

  return current;
}

/**
 * Apply JSON patches to an object following RFC 6902.
 * Includes edge case handling for string concatenation.
 */
export function applyJsonPatches(
  target: Record<string, unknown>,
  patches: JsonPatchOp[]
): Record<string, unknown> {
  let result = JSON.parse(JSON.stringify(target)); // Deep clone

  for (const patch of patches) {
    // Check for string concatenation edge case before processing
    // LLM sometimes tries to append to a string using /- syntax
    if (patch.op === "add" && patch.path.endsWith("/-")) {
      const basePath = patch.path.slice(0, -2);
      const existing = resolvePointer(result, basePath);
      if (typeof existing === "string") {
        // Convert to a replace operation that concatenates
        const basePathParts = basePath.split("/").filter((p) => p !== "");
        let parent = result;
        for (let i = 0; i < basePathParts.length - 1; i++) {
          const key = basePathParts[i];
          if (key) {
            parent = parent[key] as Record<string, unknown>;
          }
        }
        const lastKey = basePathParts[basePathParts.length - 1];
        if (lastKey) {
          parent[lastKey] = existing + String(patch.value);
        } else {
          // Root level string (unlikely but handle it)
          result = (existing + String(patch.value)) as unknown as Record<
            string,
            unknown
          >;
        }
        continue;
      }
    }

    const pathParts = patch.path.split("/").filter((p) => p !== "");

    switch (patch.op) {
      case "add":
      case "replace": {
        let current = result;
        for (let i = 0; i < pathParts.length - 1; i++) {
          const key = pathParts[i];
          if (key && !(key in current)) {
            // Check if next key is a number (array index)
            const nextKey = pathParts[i + 1];
            if (nextKey && !isNaN(Number(nextKey))) {
              current[key] = [];
            } else {
              current[key] = {};
            }
          }
          if (key) {
            current = current[key] as Record<string, unknown>;
          }
        }
        const lastKey = pathParts[pathParts.length - 1];
        if (lastKey === "-" && Array.isArray(current)) {
          // Append to array
          current.push(patch.value);
        } else if (lastKey) {
          current[lastKey] = patch.value;
        }
        break;
      }
      case "remove": {
        let current = result;
        for (let i = 0; i < pathParts.length - 1; i++) {
          const key = pathParts[i];
          if (key && current) {
            current = current[key] as Record<string, unknown>;
          }
          if (!current) break;
        }
        const lastKey = pathParts[pathParts.length - 1];
        if (current && lastKey) {
          if (Array.isArray(current)) {
            const index = parseInt(lastKey, 10);
            if (!isNaN(index)) {
              current.splice(index, 1);
            }
          } else {
            delete current[lastKey];
          }
        }
        break;
      }
      case "move": {
        if (!patch.from) break;
        const fromParts = patch.from.split("/").filter((p) => p !== "");

        // Get value from source
        let source = result;
        for (let i = 0; i < fromParts.length - 1; i++) {
          const key = fromParts[i];
          if (key) source = source[key] as Record<string, unknown>;
        }
        const sourceKey = fromParts[fromParts.length - 1];
        const value = sourceKey ? source[sourceKey] : undefined;

        // Remove from source
        if (sourceKey) {
          if (Array.isArray(source)) {
            const index = parseInt(sourceKey, 10);
            if (!isNaN(index)) source.splice(index, 1);
          } else {
            delete source[sourceKey];
          }
        }

        // Add to destination
        let dest = result;
        for (let i = 0; i < pathParts.length - 1; i++) {
          const key = pathParts[i];
          if (key) dest = dest[key] as Record<string, unknown>;
        }
        const destKey = pathParts[pathParts.length - 1];
        if (destKey) dest[destKey] = value;
        break;
      }
      case "copy": {
        if (!patch.from) break;
        const fromParts = patch.from.split("/").filter((p) => p !== "");

        // Get value from source
        let source = result;
        for (let i = 0; i < fromParts.length - 1; i++) {
          const key = fromParts[i];
          if (key) source = source[key] as Record<string, unknown>;
        }
        const sourceKey = fromParts[fromParts.length - 1];
        const value = sourceKey
          ? JSON.parse(JSON.stringify(source[sourceKey]))
          : undefined;

        // Add to destination
        let dest = result;
        for (let i = 0; i < pathParts.length - 1; i++) {
          const key = pathParts[i];
          if (key) dest = dest[key] as Record<string, unknown>;
        }
        const destKey = pathParts[pathParts.length - 1];
        if (destKey) dest[destKey] = value;
        break;
      }
      case "test": {
        let current = result;
        for (let i = 0; i < pathParts.length - 1; i++) {
          const key = pathParts[i];
          if (key) current = current[key] as Record<string, unknown>;
        }
        const lastKey = pathParts[pathParts.length - 1];
        const actual = lastKey ? current[lastKey] : undefined;
        if (JSON.stringify(actual) !== JSON.stringify(patch.value)) {
          throw new Error(
            `Test failed: expected ${JSON.stringify(patch.value)}, got ${JSON.stringify(actual)}`
          );
        }
        break;
      }
    }
  }

  return result;
}

/**
 * Ensure patches is a valid array of patch operations.
 */
export function ensurePatches(args: Record<string, unknown>): JsonPatchOp[] {
  const patches = args.patches;

  if (Array.isArray(patches)) {
    return patches as JsonPatchOp[];
  }

  if (typeof patches === "string") {
    try {
      const parsed = JSON.parse(patches);
      if (Array.isArray(parsed)) {
        return parsed as JsonPatchOp[];
      }
    } catch {
      // Try to extract array from string
      const match = patches.match(/\[[\s\S]*\]/);
      if (match) {
        try {
          const parsed = JSON.parse(match[0]);
          if (Array.isArray(parsed)) {
            return parsed as JsonPatchOp[];
          }
        } catch {
          // Fall through
        }
      }
    }
  }

  return [];
}
