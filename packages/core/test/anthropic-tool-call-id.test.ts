import assert from "node:assert/strict";
import test from "node:test";
import { AnthropicTransformer } from "../src/transformer/anthropic.transformer";

const logger = {
  debug() {},
  error() {},
};

const context = { req: { id: "test" } };

const openAIResponse = (id: string) => ({
  id: "response",
  choices: [
    {
      finish_reason: "tool_calls",
      index: 0,
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id,
            type: "function",
            function: { name: "Bash", arguments: '{"command":"true"}' },
          },
        ],
      },
    },
  ],
  created: 0,
  model: "kimi",
  object: "chat.completion",
});

const openAIStream = (id: string) => {
  const chunks = [
    {
      id: "response",
      model: "kimi",
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id,
                type: "function",
                function: { name: "Bash", arguments: "" },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    },
    {
      id: "response",
      model: "kimi",
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                function: { arguments: '{"command":"true"}' },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    },
    {
      id: "response",
      model: "kimi",
      choices: [
        { index: 0, delta: {}, finish_reason: "tool_calls" },
      ],
    },
  ];

  const body = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("");
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
};

const streamingToolCallId = async (transformer: AnthropicTransformer) => {
  const stream = await (transformer as any).convertOpenAIStreamToAnthropic(
    openAIStream("Bash_0"),
    context
  );
  const output = await new Response(stream).text();
  const startLine = output
    .split("\n")
    .find((line) => line.includes('"type":"tool_use"'));
  assert.ok(startLine);
  return JSON.parse(startLine.slice("data: ".length)).content_block.id as string;
};

const streamingUsage = async (chunks: Record<string, any>[]) => {
  const body = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("");
  const source = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
  const transformer = new AnthropicTransformer();
  transformer.logger = logger;
  const stream = await (transformer as any).convertOpenAIStreamToAnthropic(
    source,
    context
  );
  const output = await new Response(stream).text();
  const deltaLine = output
    .split("\n")
    .find((line) => line.includes('"type":"message_delta"'));
  assert.ok(deltaLine);
  return JSON.parse(deltaLine.slice("data: ".length)).usage;
};

test("non-streaming tool call IDs are unique across turns", () => {
  const transformer = new AnthropicTransformer();
  transformer.logger = logger;

  const first = (transformer as any).convertOpenAIResponseToAnthropic(
    openAIResponse("Bash_0"),
    context
  );
  const second = (transformer as any).convertOpenAIResponseToAnthropic(
    openAIResponse("Bash_0"),
    context
  );

  assert.match(first.content[0].id, /^call_[0-9a-f-]{36}$/);
  assert.match(second.content[0].id, /^call_[0-9a-f-]{36}$/);
  assert.notEqual(first.content[0].id, second.content[0].id);
});

test("streaming tool call IDs are unique across turns", async () => {
  const transformer = new AnthropicTransformer();
  transformer.logger = logger;

  const first = await streamingToolCallId(transformer);
  const second = await streamingToolCallId(transformer);

  assert.match(first, /^call_[0-9a-f-]{36}$/);
  assert.match(second, /^call_[0-9a-f-]{36}$/);
  assert.notEqual(first, second);
});

test("rewritten IDs stay paired with tool results on the next request", async () => {
  const transformer = new AnthropicTransformer();
  const id = "call_1373e98a-4b7b-4c5f-8e49-ef8425fda868";

  const request = await transformer.transformRequestOut({
    model: "kimi",
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id,
            name: "Bash",
            input: { command: "true" },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: id,
            content: "ok",
          },
        ],
      },
    ],
  });

  assert.equal(request.messages[0].tool_calls?.[0].id, id);
  assert.equal(request.messages[1].tool_call_id, id);
});

test("streaming usage accepts Kimi's choice-level usage", async () => {
  const usage = await streamingUsage([
    {
      id: "response",
      model: "kimi",
      choices: [
        {
          index: 0,
          delta: { content: "OK" },
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

test("streaming usage preserves a buffered usage-only final chunk", async () => {
  const usage = await streamingUsage([
    {
      id: "response",
      model: "kimi",
      choices: [
        { index: 0, delta: { content: "OK" }, finish_reason: null },
      ],
    },
    {
      id: "response",
      model: "kimi",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    },
    {
      id: "response",
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
