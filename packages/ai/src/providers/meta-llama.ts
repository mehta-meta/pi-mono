import { getEnvApiKey } from "../env-api-keys.js";
import { calculateCost } from "../models.js";
import type {
	AssistantMessage,
	Context,
	Message,
	Model,
	SimpleStreamOptions,
	StopReason,
	StreamFunction,
	StreamOptions,
	TextContent,
	Tool,
	ToolCall,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { buildBaseOptions } from "./simple-options.js";

// Retry configuration for rate limits
const MAX_RETRIES = 3;
const BASE_DELAY_MS = 2000; // Start with 2s delay for Llama API rate limits

/**
 * Sleep for a given number of milliseconds, respecting abort signal.
 */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new Error("Request was aborted"));
			return;
		}
		const timer = setTimeout(resolve, ms);
		signal?.addEventListener("abort", () => {
			clearTimeout(timer);
			reject(new Error("Request was aborted"));
		});
	});
}

/**
 * Check if an error is retryable (rate limit or server error)
 */
function isRetryableError(status: number): boolean {
	return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

// Llama API types (native format)
interface LlamaToolFunction {
	name: string;
	description?: string;
	parameters: Record<string, unknown>;
	strict?: boolean;
}

interface LlamaTool {
	type: "function";
	function: LlamaToolFunction;
}

interface LlamaMessage {
	role: "system" | "user" | "assistant" | "tool";
	content:
		| string
		| { type: "text"; text: string }
		| Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }>;
	tool_calls?: LlamaToolCall[];
	tool_call_id?: string;
	stop_reason?: "stop" | "tool_calls" | "length";
}

interface LlamaToolCall {
	id: string;
	function: {
		name: string;
		arguments: string;
	};
}

// Native Llama streaming event types
interface LlamaTextDelta {
	type: "text";
	text: string;
}

interface LlamaToolCallDelta {
	type: "tool_call";
	id?: string;
	function: {
		name?: string;
		arguments?: string;
	};
}

interface LlamaMetric {
	metric: string;
	value: number;
	unit?: string;
}

interface LlamaStreamEvent {
	event_type: "start" | "progress" | "complete" | "metrics";
	delta: LlamaTextDelta | LlamaToolCallDelta;
	stop_reason?: "stop" | "tool_calls" | "length";
	metrics?: LlamaMetric[];
}

interface LlamaStreamChunk {
	id?: string;
	event: LlamaStreamEvent;
}

// Convert pi-ai Tool to Llama format
function convertToolToLlama(tool: Tool): LlamaTool {
	const schema = tool.parameters;
	const parameters: Record<string, unknown> = {};

	if (schema && typeof schema === "object") {
		const typedSchema = schema as Record<string, unknown>;
		if (typedSchema.type === "object") {
			parameters.type = "object";
			if (typedSchema.properties) {
				parameters.properties = typedSchema.properties;
			}
			if (typedSchema.required) {
				parameters.required = typedSchema.required;
			}
			parameters.additionalProperties = false;
		} else {
			Object.assign(parameters, schema);
		}
	}

	return {
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters,
			strict: true,
		},
	};
}

// Convert pi-ai messages to Llama format
function convertMessagesToLlama(
	messages: Message[],
	systemPrompt?: string,
	modelSupportsImages: boolean = false,
): LlamaMessage[] {
	const result: LlamaMessage[] = [];

	if (systemPrompt) {
		result.push({ role: "system", content: sanitizeSurrogates(systemPrompt) });
	}

	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i];
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				result.push({ role: "user", content: sanitizeSurrogates(msg.content) });
			} else {
				const contentParts: Array<
					{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }
				> = [];
				for (const part of msg.content) {
					if (part.type === "text") {
						contentParts.push({ type: "text", text: sanitizeSurrogates(part.text) });
					} else if (part.type === "image") {
						const imageData = `data:${part.mimeType || "image/png"};base64,${part.data}`;
						contentParts.push({ type: "image_url", image_url: { url: imageData } });
					}
				}
				result.push({ role: "user", content: contentParts });
			}
		} else if (msg.role === "assistant") {
			const textParts = msg.content
				.filter((c): c is TextContent => c.type === "text")
				.map((c) => sanitizeSurrogates(c.text))
				.join("");

			const toolCalls = msg.content
				.filter((c): c is ToolCall => c.type === "toolCall")
				.map((tc) => ({
					id: tc.id,
					function: {
						name: tc.name,
						arguments: JSON.stringify(tc.arguments),
					},
				}));

			const llamaMsg: LlamaMessage = {
				role: "assistant",
				content: textParts ? { type: "text" as const, text: textParts } : { type: "text" as const, text: "" },
			};
			if (toolCalls.length > 0) {
				llamaMsg.tool_calls = toolCalls;
				llamaMsg.stop_reason = "tool_calls";
			}
			result.push(llamaMsg);
		} else if (msg.role === "toolResult") {
			// Extract text and image content
			const textContent = msg.content
				.filter((c): c is TextContent => c.type === "text")
				.map((c) => sanitizeSurrogates(c.text))
				.join("");
			const hasImages = msg.content.some((c) => c.type === "image");

			// Tool result with text (or placeholder if only images)
			result.push({
				role: "tool",
				tool_call_id: msg.toolCallId,
				content: textContent.length > 0 ? textContent : hasImages ? "(see attached image)" : "",
			});

			// Collect all consecutive tool results that have images, then add a single user message with all images
			const imageBlocks: Array<{ type: "image_url"; image_url: { url: string } }> = [];
			if (hasImages && modelSupportsImages) {
				for (const block of msg.content) {
					if (block.type === "image") {
						const imageData = `data:${block.mimeType || "image/png"};base64,${block.data}`;
						imageBlocks.push({ type: "image_url", image_url: { url: imageData } });
					}
				}
			}

			// Look ahead for consecutive toolResult messages with images
			let j = i + 1;
			while (j < messages.length && messages[j].role === "toolResult") {
				const nextMsg = messages[j];
				if (nextMsg.role !== "toolResult") break;

				// Add this tool result
				const nextTextContent = nextMsg.content
					.filter((c): c is TextContent => c.type === "text")
					.map((c) => sanitizeSurrogates(c.text))
					.join("");
				const nextHasImages = nextMsg.content.some((c) => c.type === "image");

				result.push({
					role: "tool",
					tool_call_id: nextMsg.toolCallId,
					content: nextTextContent.length > 0 ? nextTextContent : nextHasImages ? "(see attached image)" : "",
				});

				// Collect images from this tool result
				if (nextHasImages && modelSupportsImages) {
					for (const block of nextMsg.content) {
						if (block.type === "image") {
							const imageData = `data:${block.mimeType || "image/png"};base64,${block.data}`;
							imageBlocks.push({ type: "image_url", image_url: { url: imageData } });
						}
					}
				}

				j++;
			}

			// Skip the consecutive tool results we just processed
			i = j - 1;

			// If there were images, add a user message with all attached images
			if (imageBlocks.length > 0) {
				result.push({
					role: "user",
					content: [{ type: "text", text: "Attached image(s) from tool result:" }, ...imageBlocks],
				});
			}
		}
	}

	return result;
}

// Map Llama stop_reason to pi-ai StopReason
function mapStopReason(reason: string | undefined, hasToolCalls: boolean): StopReason {
	if (hasToolCalls || reason === "tool_calls") return "toolUse";
	if (reason === "length") return "length";
	if (reason === "stop") return "stop";
	return "stop";
}

export interface MetaLlamaOptions extends StreamOptions {
	toolChoice?: "auto" | "none" | "required";
}

export const streamMetaLlama: StreamFunction<"meta-llama", MetaLlamaOptions> = (
	model: Model<"meta-llama">,
	context: Context,
	options?: MetaLlamaOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const apiKey = options?.apiKey || getEnvApiKey("meta-llama") || "";
			const modelSupportsImages = model.input.includes("image");
			const messages = convertMessagesToLlama(context.messages, context.systemPrompt, modelSupportsImages);
			const tools = context.tools?.map(convertToolToLlama);

			const payload: Record<string, unknown> = {
				model: model.id,
				messages,
				stream: true,
			};

			if (tools && tools.length > 0) {
				payload.tools = tools;
				if (options?.toolChoice) {
					payload.tool_choice = options.toolChoice;
				}
			}

			if (options?.temperature !== undefined) {
				payload.temperature = options.temperature;
			}
			if (options?.maxTokens !== undefined) {
				payload.max_completion_tokens = options.maxTokens;
			}

			options?.onPayload?.(payload);

			// Fetch with retry logic for rate limits
			let response: Response | undefined;
			let lastError: Error | undefined;

			for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
				if (options?.signal?.aborted) {
					throw new Error("Request was aborted");
				}

				try {
					response = await fetch(`${model.baseUrl}/chat/completions`, {
						method: "POST",
						headers: {
							"Content-Type": "application/json",
							Authorization: `Bearer ${apiKey}`,
							...options?.headers,
						},
						body: JSON.stringify(payload),
						signal: options?.signal,
					});

					if (response.ok) {
						break; // Success, exit retry loop
					}

					const errorText = await response.text();

					// Check if retryable (rate limit or server error)
					if (attempt < MAX_RETRIES && isRetryableError(response.status)) {
						// Use exponential backoff
						const delayMs = BASE_DELAY_MS * 2 ** attempt;
						await sleep(delayMs, options?.signal);
						continue;
					}

					// Not retryable or max retries exceeded
					throw new Error(`Llama API error ${response.status}: ${errorText}`);
				} catch (error) {
					// Check for abort
					if (error instanceof Error) {
						if (
							error.name === "AbortError" ||
							error.message === "Request was aborted" ||
							error.message?.includes("aborted")
						) {
							throw new Error("Request was aborted");
						}
					}
					lastError = error instanceof Error ? error : new Error(String(error));
					// Network errors are retryable
					if (attempt < MAX_RETRIES) {
						const delayMs = BASE_DELAY_MS * 2 ** attempt;
						await sleep(delayMs, options?.signal);
						continue;
					}
					throw lastError;
				}
			}

			if (!response || !response.ok) {
				throw lastError ?? new Error("Failed to get response after retries");
			}

			stream.push({ type: "start", partial: output });

			const reader = response.body?.getReader();
			if (!reader) throw new Error("No response body");

			const decoder = new TextDecoder();
			let buffer = "";
			let currentTextBlock: TextContent | null = null;
			let textBlockIndex = -1;

			// Track tool calls being built up incrementally
			// The Llama API only sends the tool call id on the first delta, so we need to track the current one
			let currentToolCallId: string | null = null;
			const toolCallsInProgress: Map<string, { id: string; name: string; arguments: string; contentIndex: number }> =
				new Map();

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";

				for (const line of lines) {
					if (!line.startsWith("data: ")) continue;
					const data = line.slice(6).trim();
					if (data === "[DONE]" || !data) continue;

					try {
						const chunk = JSON.parse(data) as LlamaStreamChunk;
						const event = chunk.event;

						if (!event) continue;

						// Handle text delta
						if (event.delta?.type === "text" && event.delta.text) {
							if (!currentTextBlock) {
								currentTextBlock = { type: "text", text: "" };
								output.content.push(currentTextBlock);
								textBlockIndex = output.content.length - 1;
								stream.push({ type: "text_start", contentIndex: textBlockIndex, partial: output });
							}

							currentTextBlock.text += event.delta.text;
							stream.push({
								type: "text_delta",
								contentIndex: textBlockIndex,
								delta: event.delta.text,
								partial: output,
							});
						}

						// Handle tool call delta
						if (event.delta?.type === "tool_call") {
							const toolDelta = event.delta as LlamaToolCallDelta;
							// Use the id from the delta if present, otherwise use the current tool call id
							const callId = toolDelta.id || currentToolCallId || "";

							if (toolDelta.id) {
								// New tool call starting (id is only sent on first delta)
								currentToolCallId = toolDelta.id;
								const toolCall: ToolCall = {
									type: "toolCall",
									id: toolDelta.id,
									name: toolDelta.function?.name || "",
									arguments: {},
								};
								output.content.push(toolCall);
								const contentIndex = output.content.length - 1;

								toolCallsInProgress.set(toolDelta.id, {
									id: toolDelta.id,
									name: toolDelta.function?.name || "",
									arguments: toolDelta.function?.arguments || "",
									contentIndex,
								});

								stream.push({ type: "toolcall_start", contentIndex, partial: output });

								if (toolDelta.function?.arguments) {
									stream.push({
										type: "toolcall_delta",
										contentIndex,
										delta: toolDelta.function.arguments,
										partial: output,
									});
								}
							} else if (callId && toolCallsInProgress.has(callId)) {
								// Continuing existing tool call (appending arguments)
								const tc = toolCallsInProgress.get(callId)!;

								if (toolDelta.function?.name) {
									tc.name = toolDelta.function.name;
									(output.content[tc.contentIndex] as ToolCall).name = tc.name;
								}

								if (toolDelta.function?.arguments) {
									tc.arguments += toolDelta.function.arguments;
									stream.push({
										type: "toolcall_delta",
										contentIndex: tc.contentIndex,
										delta: toolDelta.function.arguments,
										partial: output,
									});
								}
							}
						}

						// Handle metrics (can come during progress or complete events)
						if (event.metrics) {
							for (const m of event.metrics) {
								if (m.metric === "num_prompt_tokens") {
									output.usage.input = m.value;
								} else if (m.metric === "num_completion_tokens" || m.metric === "num_generated_tokens") {
									output.usage.output = m.value;
								} else if (m.metric === "num_total_tokens") {
									output.usage.totalTokens = m.value;
								}
							}
							output.usage.cost = calculateCost(model, output.usage);
						}

						// Handle completion
						if (event.event_type === "complete" || event.stop_reason) {
							// Finalize text block
							if (currentTextBlock) {
								stream.push({
									type: "text_end",
									contentIndex: textBlockIndex,
									content: currentTextBlock.text,
									partial: output,
								});
								currentTextBlock = null;
							}

							// Finalize tool calls
							for (const [, tc] of toolCallsInProgress) {
								const existingToolCall = output.content[tc.contentIndex] as ToolCall;
								try {
									existingToolCall.arguments = JSON.parse(tc.arguments || "{}");
								} catch {
									existingToolCall.arguments = {};
								}
								stream.push({
									type: "toolcall_end",
									contentIndex: tc.contentIndex,
									toolCall: existingToolCall,
									partial: output,
								});
							}

							// Set stop reason
							const hasToolCalls = toolCallsInProgress.size > 0;
							output.stopReason = mapStopReason(event.stop_reason, hasToolCalls);
						}
					} catch {
						// Ignore JSON parse errors for incomplete chunks
					}
				}
			}

			// Ensure text block is finalized if stream ended without complete event
			if (currentTextBlock) {
				stream.push({
					type: "text_end",
					contentIndex: textBlockIndex,
					content: currentTextBlock.text,
					partial: output,
				});
			}

			// Finalize any remaining tool calls
			for (const [, tc] of toolCallsInProgress) {
				const existingToolCall = output.content[tc.contentIndex] as ToolCall;
				if (
					typeof existingToolCall.arguments === "object" &&
					Object.keys(existingToolCall.arguments).length === 0
				) {
					try {
						existingToolCall.arguments = JSON.parse(tc.arguments || "{}");
					} catch {
						existingToolCall.arguments = {};
					}
					stream.push({
						type: "toolcall_end",
						contentIndex: tc.contentIndex,
						toolCall: existingToolCall,
						partial: output,
					});
				}
			}

			// Calculate totalTokens if not provided
			if (output.usage.totalTokens === 0 && (output.usage.input > 0 || output.usage.output > 0)) {
				output.usage.totalTokens = output.usage.input + output.usage.output;
			}

			const doneReason =
				output.stopReason === "error" || output.stopReason === "aborted" ? "stop" : output.stopReason;
			stream.push({ type: "done", reason: doneReason, message: output });
		} catch (error: unknown) {
			const err = error as Error & { name?: string };
			// Check signal first (most reliable), then error message/name
			if (
				options?.signal?.aborted ||
				err.name === "AbortError" ||
				err.message === "Request was aborted" ||
				err.message?.includes("aborted")
			) {
				output.stopReason = "aborted";
				stream.push({ type: "error", reason: "aborted", error: output });
			} else {
				output.stopReason = "error";
				output.errorMessage = err.message;
				stream.push({ type: "error", reason: "error", error: output });
			}
		}
	})();

	return stream;
};

export const streamSimpleMetaLlama: StreamFunction<"meta-llama", SimpleStreamOptions> = (
	model: Model<"meta-llama">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	const baseOptions = buildBaseOptions(model, options);
	return streamMetaLlama(model, context, baseOptions);
};
