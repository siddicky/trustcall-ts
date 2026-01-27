/**
 * TrustCallJS - Utilities for validated tool calling and extraction with retries using LLMs.
 *
 * This module provides functionality for creating extractors that can generate,
 * validate, and correct structured outputs from language models. It supports
 * patch-based extraction for efficient and accurate updates to existing schemas.
 *
 * Ported from Python trustcall library by @hinthornw
 */

export {
  createExtractor,
  type ExtractionInputs,
  type ExtractionOutputs,
  type Extractor,
  type ExtractorOptions,
  type MessageDict,
} from "./extractor.js";

export {
  ValidationNode,
  type ValidationNodeOptions,
} from "./validation-node.js";

export {
  type SchemaInstance,
  type ExistingType,
  type ToolType,
} from "./types.js";
