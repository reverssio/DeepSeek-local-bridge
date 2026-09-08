# Protocol Remediation — Reasoning, Tool Calls, Session Reuse

This pass fixed four behavioral problems without touching the working
networking/authentication layer (verified by regression at the end).

## 1. Reasoning (upstream issue #9, verified locally)

Root cause: DeepSeek streams THINK and RESPONSE *fragments* through the SAME
wire path (`response/fragments/-1/content`); the THINK→RESPONSE transition is
signalled by a *fragment-array APPEND* frame the old parser ignored, and the
new fragment's initial content was dropped ("eaten first token").

Fix: `deepseek/sse.py` — fragment-aware parser tracking the fragment LIST.
Events are typed: `("reasoning", delta)` / `("content", delta)`.
`server/openai_format.py` emits `reasoning_content` (the field DeepSeek's own
official API uses) in both full responses and streaming deltas, keeping
reasoning separate from content. Verified: content "42", reasoning separate,
streaming deltas split correctly, no first-token loss.

## 2+3. Tool calls / markup leaks (reproduced via request capture)

Root cause: the bridge's schema had NO `tools` field, so the OpenAI `tools`
array OpenCode sends (bash, edit, glob, ... 11 tools) was silently dropped by
pydantic. DeepSeek never saw tool definitions, OpenCode's system prompt
contains no tool syntax, so the model improvised DSML/<bash>/fenced-markup —
leaking as plain text, never executed. The "hallucinated success" was the
model continuing a conversation in which no real tool result ever arrived.

Fix (server/tools_bridge.py):
  - `tools` now accepted in the schema and rendered into explicit
    [AVAILABLE TOOLS] + [TOOL CALLING INSTRUCTION] prompt blocks (verified
    against live upstream: the model complies with the instructed format).
  - Model output is parsed for `<tool>{json}</tool>` (instructed), DSML
    `<｜DSML｜tool_calls>` blocks (native, incl. case-insensitive tool names),
    and `<bash>`-style tags — converted to OpenAI `tool_calls[]` with
    `finish_reason: "tool_calls"` (non-stream) and index-based streaming
    deltas (stream). Markup NEVER leaks into assistant content: the stream
    buffers while a tool-opener is pending, and truncated closers are
    repaired before parsing.
  - Tool results (role=tool messages) are formatted and sent to the SAME
    DeepSeek conversation, so the model sees its own tool request (stored
    natively by DeepSeek) plus the real result. The model can no longer
    claim success without a tool result — verified with a live OpenCode run
    that actually executed bash and reported real output, and physically
    checked the created file.

## 4. Conversation sprawl (upstream issue #11, verified locally)

Root cause: OpenCode never sends the bridge's custom `conversation_id`;
every request created a new DeepSeek conversation with the whole backlog
re-flattened into it.

Fix (server/sessions.py): bridge-side session manager.
  - Every served request stores: the normalized message list it served, the
    assistant echo (text or tool_calls) it returned, and the conversation id.
  - A request is a continuation iff its messages == stored messages +
    assistant echo (+ new trailing messages). Assistant tool-call turns are
    compared structurally (names+arguments), matching how OpenCode echoes
    them (content=null).
  - Continuations send ONLY the new information (tool results / new user
    turn) to the same DeepSeek conversation. No backlog re-upload.
  - Model switch (deepseek-expert <-> deepseek-chat) starts a NEW
    conversation (a thread's model is fixed at DeepSeek) — logged.
  - New/unmatched OpenCode sessions create a fresh conversation with full
    flattened history (correct fallback; e.g. after bridge restart).
  - Title/summarization requests (detected by system-prompt signature) get
    their own throwaway conversation which is DELETED upstream afterwards.
  - Invalid/expired conversations are detected, the mapping invalidated, and
    the request transparently re-run on a fresh conversation (full history).
  - Map persists in session/conv_map.json (git-ignored, no secrets).
    Verified: 3-turn session -> exactly 1 conversation + 1 new session ->
    1 more; conversation-reuse confirmed by id comparison and log counts.

## Explicit conversation_id still works

Direct clients may still pass `conversation_id` (documented extension); it
takes precedence over the session map.

## No limits introduced

`max_tokens`/`temperature`/`top_p` remain accepted-and-ignored; no local
input/output/history truncation was added. Upstream context/output limits are
the only limits and their errors propagate classified.

## Files

- deepseek/sse.py (NEW) — fragment-aware SSE parser
- deepseek/client.py — typed events, Reply.reasoning, create marker log
- server/schemas.py — tools/tool_choice/parallel_tool_calls accepted
- server/tools_bridge.py (NEW) — instructions, parsing, result formatting
- server/sessions.py (NEW) — session mapping + recovery + title handling
- server/openai_format.py — reasoning_content, tool_calls, first/continuation
  prompt builders
- server/api.py — orchestration: mapping, tool round trips, streaming SSE
  with tool deltas, invalid-conversation recovery
- server/config.py, ratelimit.py, app.py — unchanged
- Networking/auth (auth_android.py, adb_autoconnect, retries, binding) —
  unchanged, verified intact by regression

OpenCode provider config unchanged (still http://127.0.0.1:8000/v1); added
explicit tool/bash/edit/write permissions in opencode.json (opencode-level,
not provider-level) so tool execution isn't auto-rejected.
