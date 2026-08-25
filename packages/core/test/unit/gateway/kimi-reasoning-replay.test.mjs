import assert from "node:assert/strict";
import { fork } from "node:child_process";
import { createServer } from "node:http";
import path from "node:path";
import test from "node:test";
import { parseProvidersForTest } from "@ccr/core/config/config.ts";
import { createDefaultAppConfig } from "@ccr/core/config/default-config.ts";
import { compileCoreGatewayConfig } from "@ccr/core/gateway/core-runtime/config-compiler.ts";

const unsupportedKimiRequestFields = ["reasoning_split", "reasoning", "enable_thinking"];

test("provider config preserves native OpenAI-chat reasoning options", () => {
  const [provider] = parseProvidersForTest([{
    models: ["moonshotai/Kimi-K3"],
    name: "kimi",
    openai_chat_reasoning_split: "enabled",
    openai_chat_thinking_options: "disabled"
  }]);

  assert.equal(provider.openaiChatReasoningSplit, "enabled");
  assert.equal(provider.openaiChatThinkingOptions, "disabled");
});

test("v3 gateway replays Anthropic thinking through native Kimi compatibility settings", async (t) => {
  let capturedBody;
  const providerServer = createServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    request.on("end", () => {
      capturedBody = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        choices: [{ finish_reason: "stop", index: 0, message: { content: "ok", role: "assistant" } }],
        id: "chatcmpl_test",
        model: "moonshotai/Kimi-K3",
        object: "chat.completion",
        usage: { completion_tokens: 1, prompt_tokens: 20, total_tokens: 21 }
      }));
    });
  });
  await listenOnRandomPort(providerServer);
  t.after(() => closeServer(providerServer));

  const corePort = await reservePort();
  const config = createDefaultAppConfig();
  config.gateway.corePort = corePort;
  config.Providers = [{
    api_base_url: `http://127.0.0.1:${providerServer.address().port}/v1/chat/completions`,
    api_key: "test-key",
    id: "kimi",
    models: ["moonshotai/Kimi-K3"],
    name: "kimi",
    openaiChatReasoningSplit: "enabled",
    openaiChatThinkingOptions: "disabled",
    type: "openai_chat_completions"
  }];
  config.providerPlugins = [{
    enabled: true,
    key: "kimi-reasoning-replay",
    providerName: "kimi",
    request: { bodyRemove: unsupportedKimiRequestFields }
  }];
  const compiled = await compileCoreGatewayConfig(config, "raw-trace-token", "billing-token", "core-token");
  const provider = compiled.providers.find((item) => item.name === "kimi");
  assert.equal(provider.openaiChatReasoningSplit, "enabled");
  assert.equal(provider.openaiChatThinkingOptions, "disabled");
  assert.deepEqual(
    compiled.providerPlugins.find((item) => item.key === "kimi-reasoning-replay")?.request?.bodyRemove,
    unsupportedKimiRequestFields
  );

  const child = fork(path.resolve(".test-dist/core/runtime/gateway-bootstrap.js"), [], {
    env: { ...process.env, CCR_GATEWAY_RUNTIME_ID: "kimi-reasoning-replay-test" },
    silent: true
  });
  t.after(() => child.kill("SIGTERM"));
  child.send({
    config: compiled,
    gatewayEntry: path.resolve("node_modules/@the-next-ai/ai-gateway/dist/index.js"),
    protocolVersion: 1,
    type: "gateway:start"
  });
  await waitForGateway(corePort, child);

  const gatewayResponse = await fetch(`http://127.0.0.1:${corePort}/v1/messages`, {
    body: JSON.stringify({
      max_tokens: 4096,
      messages: [
        {
          content: [
            { signature: "sig", thinking: "prior reasoning", type: "thinking" },
            { id: "call_previous", input: { command: "pwd" }, name: "Bash", type: "tool_use" }
          ],
          role: "assistant"
        },
        {
          content: [{ content: "done", tool_use_id: "call_previous", type: "tool_result" }],
          role: "user"
        }
      ],
      model: "moonshotai/Kimi-K3",
      reasoning: { effort: "high" },
      stream: false
    }),
    headers: {
      "content-type": "application/json",
      "x-ccr-core-auth": "core-token",
      "x-target-provider": "kimi"
    },
    method: "POST"
  });
  assert.equal(gatewayResponse.status, 200, await gatewayResponse.text());
  assert.ok(capturedBody);
  assert.equal(capturedBody.max_tokens, 4096);
  for (const field of unsupportedKimiRequestFields) {
    assert.equal(capturedBody[field], undefined);
  }
  const replayedAssistant = capturedBody.messages.find((message) => message.role === "assistant");
  assert.equal(replayedAssistant.reasoning_content, "prior reasoning");
  assert.equal(replayedAssistant.tool_calls[0].id, "call_previous");
  const replayedToolResult = capturedBody.messages.find((message) => message.role === "tool");
  assert.equal(replayedToolResult.tool_call_id, "call_previous");
});

function listenOnRandomPort(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
}

async function reservePort() {
  const server = createServer();
  await listenOnRandomPort(server);
  const port = server.address().port;
  await closeServer(server);
  return port;
}

function closeServer(server) {
  return new Promise((resolve) => server.close(() => resolve()));
}

async function waitForGateway(port, child) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Gateway child exited with code ${child.exitCode}`);
    }
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return;
    } catch {
      // The child has accepted its config but has not bound the port yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error("Timed out waiting for the test gateway");
}
