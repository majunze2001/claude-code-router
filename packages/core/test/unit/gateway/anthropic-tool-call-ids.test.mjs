import assert from "node:assert/strict";
import { Readable } from "node:stream";
import test from "node:test";
import {
  rewriteAnthropicJsonToolCallIdsForTest,
  rewriteAnthropicSseToolCallIdForTest,
  rewriteAnthropicToolCallIdsStream,
  shouldRewriteAnthropicToolCallIds
} from "@ccr/core/gateway/features/anthropic-tool-call-ids.ts";

async function streamText(stream) {
  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

test("rewrites repeated OpenAI tool call ids in Anthropic JSON responses", () => {
  const output = rewriteAnthropicJsonToolCallIdsForTest({
    content: [
      { id: "Bash_0", input: { command: "pwd" }, name: "Bash", type: "tool_use" },
      { text: "keep", type: "text" },
      { id: "Bash_0", input: { command: "ls" }, name: "Bash", type: "tool_use" }
    ],
    type: "message"
  });

  const toolIds = output.content.filter((block) => block.type === "tool_use").map((block) => block.id);
  assert.equal(new Set(toolIds).size, 2);
  assert.ok(toolIds.every((id) => /^call_[0-9a-f-]{36}$/.test(id)));
  assert.equal(output.content[1].text, "keep");
});

test("rewrites Anthropic streaming tool ids while preserving unrelated SSE blocks", async () => {
  const toolBlock = 'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"Bash_0","name":"Bash","input":{}}}\n\n';
  const textBlock = 'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello"}}\n\n';
  const output = await streamText(rewriteAnthropicToolCallIdsStream(
    Readable.from([toolBlock.slice(0, 41), toolBlock.slice(41), textBlock, "data: [DONE]\n\n"]),
    "text/event-stream; charset=utf-8"
  ));

  assert.doesNotMatch(output, /"id":"Bash_0"/);
  assert.match(output, /"id":"call_[0-9a-f-]{36}"/);
  assert.match(output, /"text":"hello"/);
  assert.match(output, /data: \[DONE\]\n\n$/);
});

test("leaves malformed and non-tool SSE blocks unchanged", () => {
  assert.equal(rewriteAnthropicSseToolCallIdForTest("data: not-json"), "data: not-json");
  const block = 'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"ok"}}';
  assert.equal(rewriteAnthropicSseToolCallIdForTest(block), block);
});

test("activates only for Anthropic responses converted from OpenAI chat", () => {
  assert.equal(shouldRewriteAnthropicToolCallIds({
    clientProtocol: "anthropic_messages",
    contentType: "text/event-stream",
    providerProtocol: "openai_chat_completions"
  }), true);
  assert.equal(shouldRewriteAnthropicToolCallIds({
    clientProtocol: "anthropic_messages",
    contentType: "application/json",
    providerProtocol: "openai_chat_completions"
  }), true);
  assert.equal(shouldRewriteAnthropicToolCallIds({
    clientProtocol: "anthropic_messages",
    contentType: "text/event-stream",
    providerProtocol: "anthropic_messages"
  }), false);
  assert.equal(shouldRewriteAnthropicToolCallIds({
    clientProtocol: "openai_chat_completions",
    contentType: "application/json",
    providerProtocol: "openai_chat_completions"
  }), false);
});
