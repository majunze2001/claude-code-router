import { randomUUID } from "node:crypto";
import { Readable, Transform } from "node:stream";
import { StringDecoder } from "node:string_decoder";
import type { GatewayProviderProtocol } from "@ccr/core/contracts/app";
import { isRecord, stringValue } from "@ccr/core/gateway/internal/value";

export function shouldRewriteAnthropicToolCallIds(input: {
  clientProtocol: GatewayProviderProtocol | undefined;
  contentType: string | undefined;
  providerProtocol: GatewayProviderProtocol | undefined;
}): boolean {
  const contentType = input.contentType?.toLowerCase() ?? "";
  return input.clientProtocol === "anthropic_messages" &&
    input.providerProtocol === "openai_chat_completions" &&
    (contentType.includes("application/json") || contentType.includes("text/event-stream"));
}

export function rewriteAnthropicToolCallIdsStream(
  input: Readable,
  contentType: string | undefined
): Readable {
  return contentType?.toLowerCase().includes("text/event-stream")
    ? rewriteAnthropicToolCallIdsSseStream(input)
    : rewriteAnthropicToolCallIdsJsonStream(input);
}

function rewriteAnthropicToolCallIdsJsonStream(input: Readable): Readable {
  const chunks: Buffer[] = [];
  return input.pipe(new Transform({
    transform(chunk, _encoding, callback) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      callback();
    },
    flush(callback) {
      const body = Buffer.concat(chunks);
      try {
        const parsed = JSON.parse(body.toString("utf8")) as unknown;
        this.push(JSON.stringify(rewriteAnthropicJsonToolCallIds(parsed)));
      } catch {
        this.push(body);
      }
      callback();
    }
  }));
}

function rewriteAnthropicToolCallIdsSseStream(input: Readable): Readable {
  const decoder = new StringDecoder("utf8");
  let pending = "";
  return input.pipe(new Transform({
    transform(chunk, _encoding, callback) {
      pending += decoder.write(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      pending = drainAnthropicSseBlocks(this, pending, false);
      callback();
    },
    flush(callback) {
      pending += decoder.end();
      drainAnthropicSseBlocks(this, pending, true);
      pending = "";
      callback();
    }
  }));
}

function drainAnthropicSseBlocks(stream: Transform, text: string, flush: boolean): string {
  let cursor = 0;
  for (const match of text.matchAll(/\r?\n\r?\n/g)) {
    const index = match.index ?? 0;
    const delimiter = match[0];
    const block = text.slice(cursor, index);
    cursor = index + delimiter.length;
    stream.push(`${rewriteAnthropicSseToolCallId(block)}${delimiter}`);
  }

  const trailing = text.slice(cursor);
  if (!flush) {
    return trailing;
  }
  if (trailing) {
    stream.push(rewriteAnthropicSseToolCallId(trailing));
  }
  return "";
}

function rewriteAnthropicSseToolCallId(block: string): string {
  if (!block.trim()) {
    return block;
  }
  const parsed = parseSseJsonData(block);
  if (
    !isRecord(parsed) ||
    stringValue(parsed.type) !== "content_block_start" ||
    !isRecord(parsed.content_block) ||
    stringValue(parsed.content_block.type) !== "tool_use"
  ) {
    return block;
  }
  return replaceSseDataLines(block, JSON.stringify({
    ...parsed,
    content_block: {
      ...parsed.content_block,
      id: freshToolCallId()
    }
  }));
}

function rewriteAnthropicJsonToolCallIds(payload: unknown): unknown {
  if (!isRecord(payload) || !Array.isArray(payload.content)) {
    return payload;
  }
  let changed = false;
  const content = payload.content.map((block) => {
    if (!isRecord(block) || stringValue(block.type) !== "tool_use") {
      return block;
    }
    changed = true;
    return {
      ...block,
      id: freshToolCallId()
    };
  });
  return changed ? { ...payload, content } : payload;
}

function freshToolCallId(): string {
  return `call_${randomUUID()}`;
}

function parseSseJsonData(block: string): unknown {
  const data = block
    .split(/\r?\n/g)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data || data === "[DONE]") {
    return undefined;
  }
  try {
    return JSON.parse(data) as unknown;
  } catch {
    return undefined;
  }
}

function replaceSseDataLines(block: string, data: string): string {
  const newline = block.includes("\r\n") ? "\r\n" : "\n";
  const lines = block.split(/\r?\n/g);
  const output: string[] = [];
  let replaced = false;
  for (const line of lines) {
    if (!line.startsWith("data:")) {
      output.push(line);
      continue;
    }
    if (!replaced) {
      output.push(`data: ${data}`);
      replaced = true;
    }
  }
  return output.join(newline);
}

export function rewriteAnthropicJsonToolCallIdsForTest(payload: unknown): unknown {
  return rewriteAnthropicJsonToolCallIds(payload);
}

export function rewriteAnthropicSseToolCallIdForTest(block: string): string {
  return rewriteAnthropicSseToolCallId(block);
}
