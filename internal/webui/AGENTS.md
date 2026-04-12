# WebUI Package AGENTS.md

This file is package-local guidance for `internal/webui`. It is intentionally narrower than the repo-root Mantle `AGENTS.md` and should be read together with it, not instead of it.

## Package Scope

`internal/webui` is the embedded browser client for Mantle's local server mode.

This package is responsible for:
- embedding static assets into the Go binary via [webui.go](/work/apps/mantle/internal/webui/webui.go)
- delivering the browser application shell from `serve` mode through `webui.StaticFS()`
- providing a thin UI over the Responses API and model listing API

This package is not responsible for:
- defining API semantics
- implementing inference or response storage
- adding new backend-only features without a matching UI purpose

Keep the package thin. If a change is generic server behavior, it likely belongs in `internal/api` or `cmd/mantle/serve.go`, not here.

## Current Layout

- [webui.go](/work/apps/mantle/internal/webui/webui.go): embed boundary; returns an `http.FileSystem` rooted at `static/`
- [static/index.html](/work/apps/mantle/internal/webui/static/index.html): document shell, import map, stylesheet/module entrypoints
- [static/app.js](/work/apps/mantle/internal/webui/static/app.js): entire client application state, fetch flow, SSE parsing, rendering
- [static/style.css](/work/apps/mantle/internal/webui/static/style.css): all styling and responsive layout

The actual routes are registered outside this package in [cmd/mantle/serve.go](/work/apps/mantle/cmd/mantle/serve.go), which:
- mounts `/static/*` from `webui.StaticFS()`
- serves `index.html` at `/`
- falls back to `index.html` for unknown non-API GET routes

## Working Rules

- Preserve the package's zero-build nature unless the user explicitly asks to introduce a frontend toolchain.
- Prefer browser-native APIs and simple modules over bundlers, transpilers, or framework-heavy additions.
- Treat the embedded UI as part of the shipped binary. Avoid solutions that quietly turn it into a network-dependent dev shell.
- Keep backend contracts explicit. Do not guess undocumented event shapes or API fields in the client.
- If the UI begins to exercise more of the Responses API, centralize protocol parsing instead of scattering event handling across the component tree.
- Avoid adding package-specific dependencies unless explicitly requested. The current design relies on CDN ESM modules; replacing that with vendored assets is acceptable, adding more remote dependencies is not.

## Change Discipline

When editing this package:
- verify whether the change belongs in `internal/webui` or in the server/router layer first
- keep `webui.go` trivial; do not add app logic there
- keep HTML declarative; put state and event logic in `app.js`
- keep CSS tokenized with the existing custom properties before introducing one-off colors or spacing values
- prefer small helper functions over inlining more protocol parsing into the render path

## Validation Expectations

There is no package-local automated coverage today, so validate deliberately.

Minimum checks after non-trivial changes:
- `gofmt ./...`
- `GOEXPERIMENT=simd go test ./internal/webui ./cmd/mantle ./internal/api`
- `GOEXPERIMENT=simd go test ./...` for broader UI/API contract changes
- `GOEXPERIMENT=simd go build -o bin/mantle ./cmd/mantle`
- manual browser smoke test of `mantle serve` with the affected UI flow

For UI-affecting changes, manually verify:
- initial page load
- model list refresh
- sync response path
- streaming response path
- stream cancellation path
- mobile-width layout

## Known Findings

These are current observations from the code and should be treated as real cleanup candidates, not generic wishlist items.

### Missing Coverage

- There are no tests in `internal/webui` at all. Nothing verifies that [webui.go](/work/apps/mantle/internal/webui/webui.go) still exposes the expected files, that `index.html` is present, or that the client-facing assumptions still match the backend event stream.
- The SPA serving path is implemented in [serve.go](/work/apps/mantle/cmd/mantle/serve.go), but there is no smoke coverage for `/`, `/static/*`, and the non-API fallback behavior together.

### Embedded UI Is Not Actually Self-Contained

- [static/index.html](/work/apps/mantle/internal/webui/static/index.html) loads Google Fonts and `esm.sh` modules over the network.
- That means the "embedded web UI" still depends on third-party CDNs at runtime. In offline, air-gapped, or privacy-sensitive environments, the UI shell may render incorrectly or fail to boot entirely.
- This is the highest-priority architectural gap in the package.

### Feature Surface Does Not Match Backend Surface

- The client only uses `/v1/models` and `/v1/responses`.
- The backend also exposes response retrieval, deletion, cancellation, input item inspection, compaction, input token counting, and chat completions in [internal/api/responses.go](/work/apps/mantle/internal/api/responses.go) and [internal/api/chat_completions.go](/work/apps/mantle/internal/api/chat_completions.go).
- Some of that omission is a scope choice, but today the UI gives no affordance for inspecting stored responses, resuming from known IDs, or debugging request history even though the server supports it.

### Misleading Or Fragile UI Semantics

- `chars ${tokenCount}` in [static/app.js](/work/apps/mantle/internal/webui/static/app.js) is a character count, not a token count. The label is not technically wrong because it says `chars`, but it sits beside true token usage and invites misreading as a token proxy.
- The footer text labels `previousResponseID` as `thread id`, but it is actually the last response ID used for `previous_response_id` chaining. That should be renamed in the UI.
- `refreshModels()` collapses every failure mode into `API unreachable`, which hides useful distinctions such as malformed JSON, permission issues, or backend 500s.

### Probable Stale / Overlapping Reasoning Rendering

- [static/app.js](/work/apps/mantle/internal/webui/static/app.js) supports two reasoning display paths:
  - parsing inline `<think>...</think>` blocks out of normal output text
  - rendering separate `thinking` state from `response.output_reasoning.*` events
- That overlap is useful for compatibility, but it also means the UI can duplicate reasoning if both mechanisms are populated for the same response.
- This is a good candidate for cleanup into one canonical rendering strategy plus one clearly-marked fallback.

### Session State Is Too Ephemeral

- The only persisted client preference is `thinkMode` in `localStorage`.
- Messages, selected model, generation settings, and the current `previous_response_id` chain are lost on refresh even though the backend maintains response records.
- For a local console, that makes reloads feel more destructive than they need to be.

## Cleanup Targets

If the user asks for cleanup in this package, start here:

1. Remove runtime CDN dependence by vendoring or embedding frontend dependencies and fonts locally.
2. Add package-level tests for embedded assets and a server-level smoke test for `/`, `/static/app.js`, and SPA fallback behavior.
3. Split `static/app.js` into small protocol/state/render modules if the file keeps growing; it is already the entire application in one file.
4. Consolidate reasoning rendering so inline `<think>` parsing and structured reasoning events do not fight each other.
5. Rename UI text around `previousResponseID` to reflect actual semantics.
6. Improve connection/error reporting so model refresh and request failures expose actionable information.

## Improvement Opportunities

These are sensible next steps if the user wants the package improved, not just documented.

- Add a small "response inspector" panel that can fetch a saved response by ID and show input items using existing backend endpoints.
- Persist session-local controls such as selected model, temperature, and max output tokens in `localStorage`.
- Add a lightweight request history sidebar backed by stored response IDs rather than in-memory-only chat state.
- Surface backend features already present but invisible today, especially input token counting and response retrieval.
- Add a frontend smoke harness, even if minimal, so regressions in SSE parsing and page boot are caught before release.
- Reduce repeated fetch/SSE glue in `app.js` by extracting protocol helpers. The current single-file approach is still workable, but the protocol logic is now substantial enough to deserve structure.

## Project Memory

- `internal/webui` is currently a zero-build embedded frontend: one Go embed file plus static `index.html`, `app.js`, and `style.css`.
- The package presently depends on runtime CDNs for fonts and ESM modules despite being embedded; treat removal of that network dependency as a primary cleanup opportunity.
- The frontend currently covers `/v1/models` and `/v1/responses` only; richer response-store endpoints already exist in `internal/api` and are candidates for UI exposure.
