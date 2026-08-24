import assert from "node:assert/strict";
import test from "node:test";
import { AnthropicTransformer } from "../src/transformer/anthropic.transformer";

const transformWithBillingValue = async (value: string) => {
  const transformer = new AnthropicTransformer();
  return transformer.transformRequestOut({
    model: "kimi",
    system: [
      {
        type: "text",
        text: `x-anthropic-billing-header: cc_version=2.1.138; cch=${value};`,
      },
      { type: "text", text: "Stable system prompt" },
    ],
    messages: [{ role: "user", content: "Hello" }],
  });
};

test("different Claude billing values produce identical Kimi prompts", async () => {
  const first = await transformWithBillingValue("first");
  const second = await transformWithBillingValue("second");

  assert.deepEqual(first.messages, second.messages);
  assert.equal(first.messages[0].role, "system");
  assert.deepEqual(first.messages[0].content, [
    {
      type: "text",
      text: "Stable system prompt",
      cache_control: undefined,
    },
  ]);
});

test("ordinary first system blocks are preserved", async () => {
  const transformer = new AnthropicTransformer();
  const result = await transformer.transformRequestOut({
    model: "kimi",
    system: [{ type: "text", text: "Keep this prompt" }],
    messages: [],
  });

  assert.equal(result.messages[0].role, "system");
  assert.deepEqual(result.messages[0].content, [
    { type: "text", text: "Keep this prompt", cache_control: undefined },
  ]);
});
