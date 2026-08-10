import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the symbolic backend briefing", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Clifft symbolic backend - team briefing<\/title>/i);
  assert.match(html, /One week into the CPU backend refactor/);
  assert.match(html, /What are the headline results\?/);
  assert.match(html, /What techniques did SymFT introduce\?/);
  assert.match(html, /What actually drives SymFT's performance\?/);
  assert.match(html, /What does this mean for a GPU backend\?/);
  assert.match(html, />5\.09x</);
  assert.match(html, /16\.8(?:<!-- -->)?x/);
  assert.match(html, /Sampling is no longer the whole story\./);
  assert.match(html, /detector references differ/);
  assert.doesNotMatch(html, /codex-preview|loading skeleton|react-loading-skeleton/i);
});
