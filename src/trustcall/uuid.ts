/**
 * Cross-runtime UUID v4 generator.
 * Uses the standard crypto.randomUUID() API available in:
 * - Node.js 18+ (via globalThis.crypto)
 * - Deno (via globalThis.crypto)
 * - Bun (via globalThis.crypto)
 */

/**
 * Generate a UUID v4 string.
 * @returns A UUID v4 string
 */
export function uuidv4(): string {
  // Check if crypto.randomUUID is available (Node.js 18+, Deno, Bun)
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  // This should never be reached if using supported runtimes
  throw new Error(
    "crypto.randomUUID is not available. Please use Node.js 18+, Deno, or Bun."
  );
}
