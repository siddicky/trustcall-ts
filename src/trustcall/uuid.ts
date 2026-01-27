/**
 * Cross-runtime UUID v4 generator.
 * Uses the standard crypto.randomUUID() API available in:
 * - Node.js 18+ (via node:crypto)
 * - Deno (via globalThis.crypto)
 * - Bun (via globalThis.crypto)
 */

/**
 * Generate a UUID v4 string.
 * @returns A UUID v4 string
 */
export function uuidv4(): string {
  // For modern runtimes (Node.js 18+, Deno, Bun), crypto.randomUUID is available
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  // Fallback for older Node.js versions (though package.json requires Node 18+)
  // This should never be reached in practice
  throw new Error(
    "crypto.randomUUID is not available. Please use Node.js 18+, Deno, or Bun."
  );
}
