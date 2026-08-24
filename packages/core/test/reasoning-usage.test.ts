import assert from "node:assert/strict";
import test from "node:test";
import { AnthropicTransformer } from "../src/transformer/anthropic.transformer";
import { ReasoningTransformer } from "../src/transformer/reasoning.transformer";

const logger = {
  debug() {},
  error() {},
};

const context = { req: { id: "test" } };

const anthropicUsageAfterReasoning = async (chunks: Record<string, any>[]) => {
  const body =
    chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") +
    "data: [DONE]\n\n";
  const response = new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });

  const reasoning = new ReasoningTransformer();
  reasoning.logger = logger;
  const transformed = await reasoning.transformResponseOut(response);

  const anthropic = new AnthropicTransformer();
  anthropic.logger = logger;
  const stream = await (anthropic as any).convertOpenAIStreamToAnthropic(
    transformed.body,
    context
  );
  const output = await new Response(stream).text();
  const deltaLine = output
    .split("\n")
    .find((line) => line.includes('"type":"message_delta"'));
  assert.ok(deltaLine);
  return JSON.parse(deltaLine.slice("data: ".length)).usage;
};

test("reasoning transform preserves Kimi choice-level usage", async () => {
  const usage = await anthropicUsageAfterReasoning([
    {
      model: "kimi",
      choices: [
        {
          index: 0,
          delta: { reasoning_content: "Think" },
          finish_reason: null,
        },
      ],
    },
    {
      model: "kimi",
      choices: [
        { index: 0, delta: { content: "OK" }, finish_reason: null },
      ],
    },
    {
      model: "kimi",
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: "stop",
          usage: {
            prompt_tokens: 120,
            completion_tokens: 8,
            prompt_tokens_details: { cached_tokens: 20 },
          },
        },
      ],
    },
  ]);

  assert.deepEqual(usage, {
    input_tokens: 100,
    output_tokens: 8,
    cache_read_input_tokens: 20,
  });
});

test("reasoning transform preserves OpenAI usage-only chunks", async () => {
  const usage = await anthropicUsageAfterReasoning([
    {
      model: "kimi",
      choices: [
        { index: 0, delta: { content: "OK" }, finish_reason: null },
      ],
    },
    {
      model: "kimi",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    },
    {
      model: "kimi",
      choices: [],
      usage: {
        prompt_tokens: 75,
        completion_tokens: 4,
        prompt_tokens_details: { cached_tokens: 5 },
      },
    },
  ]);

  assert.deepEqual(usage, {
    input_tokens: 70,
    output_tokens: 4,
    cache_read_input_tokens: 5,
  });
});
