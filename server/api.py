"""
OpenAI-compatible FastAPI server for DeepSeek.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream=true supported; tools supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz is exempt.

Session semantics (server/sessions.py):
    requests from the same OpenCode session are matched via message-prefix
    fingerprinting and continue the SAME native DeepSeek conversation; only
    the new turn's information is sent upstream. Title/summarization requests
    get their own throwaway conversation, deleted upstream afterwards.

Tool semantics (server/tools_bridge.py):
    the OpenAI `tools` array is rendered into prompt instructions; model
    output markup (<tool>{json}</tool>, DSML, <bash> tags) is parsed back
    into structured OpenAI `tool_calls` (streaming deltas included) — never
    leaked as plain assistant text.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.client import DeepSeekClient

from .config import (
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    is_known_model,
    resolve_model_type,
)
from .openai_format import (
    build_continuation_prompt,
    build_first_prompt,
    completion_response,
    new_turn_messages,
    stream_chunks,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest
from .sessions import get_map, is_title_request
from .tools_bridge import ToolCall, parse_tool_calls

load_dotenv()

app = FastAPI(title="DeepSeek OpenAI-compatible API", version="0.2.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# One shared client (and its signed-in session) built lazily on first use.
_client: DeepSeekClient | None = None
_client_lock = threading.Lock()

# Serialized upstream turns (the PoW store isn't reentrant — upstream design).
_turn_lock = threading.Lock()


def get_client() -> DeepSeekClient:
    """Build (once) the shared client and its signed-in session.

    Session resolution: cached file → headless capture off the persistent
    profile. If neither works and SERVER_INTERACTIVE_LOGIN is on (the default),
    it opens a visible browser window so you can sign in — the triggering
    request blocks until you finish. If interactive login is off, it raises
    `LoginRequired`, which the endpoint turns into an actionable 503.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = DeepSeekClient(allow_interactive=SERVER_INTERACTIVE_LOGIN)
    return _client


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "deepseek"}
            for name in MODEL_MAP
        ],
    }


# --- upstream turn execution -------------------------------------------------

class _TurnStream:
    """Wraps a DeepSeek _Stream: typed events + tool-call parsing + finish.

    Emission policy: content is buffered while it might still turn out to be
    tool-call markup. Once markup starts, everything is held until the stream
    ends and the final parse decides what to emit (clean text, or clean text +
    tool_calls). Markup-free text streams through immediately (holding back
    only a trailing partial opener like '<' or '<tool').

    Concurrency: the ENTIRE upstream turn (including stream consumption) is
    serialized — OpenCode fires its background title request and the main
    request concurrently, and DeepSeek's backend stalls one of two parallel
    completions on the same account. The lock is held from first event to
    stream end (and released on early client aborts via generator close).
    """

    def __init__(self, upstream, tool_names, model_name, turn_lock, tools_def=None):
        self._upstream = upstream
        self._tool_names = tool_names
        self._tools_def = tools_def or tool_names
        from .tools_bridge import get_tool_nonce
        self._nonce = get_tool_nonce(tools_def) if tools_def else None
        self.finish_reason = "stop"
        self.conversation_id = None
        self._full_text: list[str] = []
        self._saw_markup = False
        self._emitted = 0
        self._turn_lock = turn_lock
        self._pending_calls: list = []
        self._emitted_calls = 0

    # Partial-opener prefixes that must be held back mid-stream: a trailing
    # '<' (or an incomplete '<tool', '<bash', ...) might become tool markup.
    _OPENERS = (
        "<tool>", "<tool ",
        "<｜｜DSML", "<｜DSML", "<||DSML", "<|DSML",
        "<bash>", "<edit>", "<read>", "<write>", "<grep>", "<glob>", "<webfetch>",
    )

    @classmethod
    def _pending_opener_len(cls, text: str, emitted: int) -> int:
        """Length of a trailing partial tool-opener, or 0 if none."""
        tail = text[emitted:]
        for opener in cls._OPENERS:
            for cut in range(1, len(opener) + 1):
                prefix = opener[:cut]
                if tail.endswith(prefix) and prefix != opener:
                    return len(prefix)
        return 0

    def events(self):
        from .tools_bridge import contains_tool_markup

        def _looks_like_markup(t: str) -> bool:
            """Complete markup OR an obviously-open markup block.

            DeepSeek sometimes ends the stream with the closing tag truncated
            (e.g. '...</tool' without '>'). If any opener exists, treat the
            whole text as markup and parse leniently.
            """
            return contains_tool_markup(t) or "<tool" in t or "DSML" in t

        # Serialize the ENTIRE upstream turn (stream consumption included):
        # OpenCode fires its background title request and the main request
        # concurrently, and DeepSeek stalls one of two parallel completions
        # on the same account. Released in `finally`, so client aborts that
        # close the generator early still free the lock.
        self._turn_lock.acquire()
        try:
            for kind, delta in self._upstream.events():
                if kind == "reasoning":
                    yield ("reasoning", delta)
                    continue
                if kind != "content":
                    continue  # e.g. ("finish", None)
                self._full_text.append(delta)
                joined = "".join(self._full_text)
                if not self._saw_markup and _looks_like_markup(joined):
                    self._saw_markup = True
                if self._saw_markup:
                    # Progressive tool-call emission: whenever the joined text
                    # contains at least one COMPLETE tool block, parse what's
                    # available and emit any newly completed calls immediately
                    # (so clients see tool deltas as soon as they close).
                    import re as _re
                    if _re.search(r"</tool\s*>|</invoke\s*>|</[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*(?:calls|tool_calls|invoke)>", joined):
                        new_calls, _clean = parse_tool_calls(joined, self._tools_def, expected_nonce=self._nonce)
                        if len(new_calls) > len(self._pending_calls):
                            self._pending_calls = new_calls
                            fresh = self._pending_calls[self._emitted_calls:]
                            self._emitted_calls = len(self._pending_calls)
                            yield ("tool_calls", fresh)
                    continue  # hold text; decide at stream end
                # no markup so far: emit everything except a trailing partial
                # opener (e.g. '<', '<to', '<tool') that might become markup
                pending = self._pending_opener_len(joined, self._emitted)
                emit_upto = len(joined) - pending
                if emit_upto > self._emitted:
                    yield ("content", joined[self._emitted:emit_upto])
                    self._emitted = emit_upto
        finally:
            self._turn_lock.release()

        text = "".join(self._full_text)
        if self._saw_markup:
            calls, clean = parse_tool_calls(text, self._tools_def, expected_nonce=self._nonce)
            # keep calls already emitted progressively (same objects by
            # parse order); emit any remaining ones now
            all_calls = calls or self._pending_calls
            if all_calls:
                self.finish_reason = "tool_calls"
                if len(all_calls) > self._emitted_calls:
                    fresh = all_calls[self._emitted_calls:]
                    self._emitted_calls = len(all_calls)
                    yield ("tool_calls", fresh)
                # clean text that preceded/followed the tool blocks
                if clean:
                    from .tools_bridge import get_unemitted_clean
                    rem = get_unemitted_clean(clean, text[:self._emitted])
                    if rem and rem.strip() not in (">", "<", ">>", "<<"):
                        yield ("content", rem)
            elif self._emitted_calls:
                # calls were already emitted progressively; nothing more to do
                self.finish_reason = "tool_calls"
            else:
                # Markup was detected but NOTHING parsed as a tool call —
                # the model attempted a tool call in a dialect we couldn't
                # structure. Emitting the (stripped) text as a normal answer
                # would let the conversation claim an action that never ran.
                # Instead surface a hard error so the agent can retry.
                yield (
                    "error",
                    "The model attempted a tool call in an unrecognized "
                    "format (raw markup suppressed). Ask it to retry using "
                    "the exact <tool>{\"name\": ..., \"arguments\": {...}}</tool> "
                    "JSON format.",
                )
        else:
            if len(text) > self._emitted:
                yield ("content", text[self._emitted:])
        self.conversation_id = self._upstream.conversation_id


def _run_turn(client, prompt, conversation_id, model_type, thinking, search,
              tool_names, model_name, tools_def=None, ref_file_ids=None):
    """Execute one upstream turn; the _TurnStream serializes consumption."""
    upstream = client.stream(
        prompt, conversation_id=conversation_id,
        model=model_type, thinking=thinking, search=search,
        ref_file_ids=ref_file_ids,
    )
    return _TurnStream(upstream, tool_names, model_name, _turn_lock, tools_def=tools_def)


_INVALID_CONV_MARKERS = (
    "chat session not exist", "chat_session", "not found", "invalid",
    "expired", "permission", "concurrent", "invalid message id",
    "biz_code=1", "biz_code=26",
)


def _looks_like_invalid_conversation(msg: str) -> bool:
    low = msg.lower()
    return any(m in low for m in _INVALID_CONV_MARKERS) and (
        "not exist" in low or "invalid" in low or "not found" in low
        or "expired" in low
    )


def _oc_session_id(request) -> str | None:
    """OpenCode's stable session id, when present.

    Verified in 1.18.27 traffic: custom (non-opencode) providers get
    'x-session-affinity: <sessionID>' and 'X-Session-Id: <sessionID>'
    headers on every chat request."""
    try:
        return (
            request.headers.get("x-opencode-session")
            or request.headers.get("x-session-affinity")
            or request.headers.get("x-session-id")
        )
    except Exception:
        return None


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest):
    # ---- forensic request table (TEMPORARY; no secrets logged) ----
    # Classifies each request: session id from headers, type, conversation
    # create/reuse. Written to logs/forensics.txt
    try:
        hdrs = request.headers
        oc_session = (hdrs.get("x-opencode-session") or hdrs.get("x-session-affinity")
                      or hdrs.get("x-session-id") or hdrs.get("session-id"))
        roles = [m.role for m in req.messages]
        tools_n = len(req.tools or [])
        tnames = [t.get("function", {}).get("name") for t in (req.tools or []) if isinstance(t, dict)]
        title = is_title_request([m.dict() for m in req.messages])
        has_tool_msgs = any(m.role == "tool" for m in req.messages)
        has_asst_tc = any(m.role == "assistant" and m.tool_calls for m in req.messages)
        BASE_DIR = Path(__file__).resolve().parent.parent
        log_file = BASE_DIR / "logs" / "forensics.txt"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] sess={oc_session or '-'} model={req.model} stream={req.stream} "
                    f"roles={roles} n={len(req.messages)} tools={tools_n} tnames={tnames} "
                    f"title={title} tool_result={has_tool_msgs} asst_tc={has_asst_tc} "
                    f"cid={req.conversation_id or '-'} "
                    f"effort={req.reasoning_effort or '-'} budget={req.thinking_budget or '-'} "
                    f"thinking={req.thinking} "
                    f"ua={hdrs.get('user-agent','-')[:40]}\n")
    except Exception:
        pass

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    title_req = is_title_request([m.dict() for m in req.messages]) or (
        request.headers.get("x-opencode-title") == "true"
    )

    # Title/summarization requests: OpenCode's ensureTitle() sends a background
    # request to summarize the conversation into a title. We generate an intelligent,
    # concise (<= 50 chars), grammatical title directly based on the user's prompt.
    # This avoids throwaway conversations on chat.deepseek.com, avoids PoW challenge
    # overhead, eliminates 503 error retries, and never pollutes or corrupts session mapping.
    if title_req:
        from .title_generator import generate_title
        title = generate_title([m.dict() for m in req.messages])
        if not req.stream:
            return completion_response(req.model, title, prompt="", conversation_id=None)

        def sse_title_stream():
            import uuid
            cid = "chatcmpl-title-" + uuid.uuid4().hex
            created = int(time.time())
            f1 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
            }
            yield f"data: {json.dumps(f1, ensure_ascii=False)}\n\n"
            f2 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": title}}],
            }
            yield f"data: {json.dumps(f2, ensure_ascii=False)}\n\n"
            f3 = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(f3, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_title_stream(), media_type="text/event-stream")

    tools = req.tools or []
    tool_names = [t.get("function", {}).get("name") for t in tools
                  if isinstance(t, dict)]

    # Effective thinking: explicit extra_body flag OR an OpenCode variant
    # passthrough (reasoning_effort / thinking_budget arrive when the user
    # picks a reasoning variant of the model).
    eff_thinking = bool(req.thinking) or req.reasoning_effort is not None \
        or req.thinking_budget is not None
    cmap = get_map()

    try:
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:  # session/login failure
        return _error(f"Failed to initialise DeepSeek session: {e}")

    msg_dicts = [m.dict() for m in req.messages]

    # OpenCode 1.18.27 sends a stable session id for custom providers via
    # the x-session-affinity / X-Session-Id headers (verified in captured
    # traffic). Use it as the PRIMARY mapping key; fall back to message
    # fingerprints when absent (e.g. curl / other clients).
    oc_session = _oc_session_id(request)
    from .sessions import get_session_lock
    session_lock = get_session_lock(oc_session)

    with session_lock:
        mapped = cmap.find_by_session(oc_session, req.model) if oc_session \
            else cmap.find(msg_dicts)

        # Explicit client-supplied conversation_id wins (documented extension).
        if req.conversation_id:
            conversation_id = req.conversation_id
            model_type = None  # resume: keep the thread's own model
            prompt_mode = "explicit"
        elif mapped:
            conversation_id = mapped["conversation_id"]
            # A thread's model is fixed at creation. Only a genuine model-name
            # change forces a new DeepSeek conversation.
            if mapped.get("model") and mapped["model"] != req.model:
                print(f"[sessions] model switch {mapped['model']} -> {req.model}: "
                      "starting a new DeepSeek conversation")
                conversation_id = None
                model_type = resolve_model_type(req.model)
                prompt_mode = "fresh"
            else:
                model_type = None
                prompt_mode = "continuation"
        else:
            conversation_id = None
            model_type = resolve_model_type(req.model)
            prompt_mode = "fresh"

    # Multimodal image extraction: scan messages for images (local Android paths,
    # data URIs, or public URLs) and upload them to DeepSeek Web
    ref_file_ids = []
    image_locations = []
    try:
        from deepseek.multimodal import process_multimodal_inputs_with_meta
        ref_file_ids, image_locations = await run_in_threadpool(process_multimodal_inputs_with_meta, client, req.messages)
    except Exception as img_err:
        print(f"[multimodal] image processing note: {img_err}", flush=True)

    attachment_hint = ""
    if image_locations:
        attachment_hint = (
            f"[ATTACHED IMAGE: The image from {', '.join(image_locations)} has been uploaded and attached to this request as ref_file_ids. "
            f"You can see its visual pixels directly in your multimodal context. "
            f"Please analyze the visual contents of the attached image directly without calling bash or file tools to inspect it.]\n\n"
        )

    if prompt_mode in ("fresh", "explicit"):
        prompt = build_first_prompt(req.messages, tools)
        if attachment_hint:
            prompt = attachment_hint + prompt
    else:
        new_msgs = new_turn_messages(req.messages)
        used_tool = bool(mapped.get("last_had_tool_calls")) if mapped else False
        if not new_msgs:
            # nothing new (e.g. duplicate retry) — treat as fresh
            prompt = build_first_prompt(req.messages, tools)
            prompt_mode = "fresh"
            model_type = resolve_model_type(req.model)
            conversation_id = None
            if attachment_hint:
                prompt = attachment_hint + prompt
        else:
            prompt = build_continuation_prompt(new_msgs, tools, used_tool, all_messages=req.messages)
            if attachment_hint and any(m.role == "user" for m in new_msgs):
                prompt = attachment_hint + prompt

    def serve(prompt, conversation_id, model_type, prompt_mode):
        # log the create/reuse decision into the forensic table
        try:
            BASE_DIR = Path(__file__).resolve().parent.parent
            log_file = BASE_DIR / "logs" / "forensics.txt"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                act = ("REUSE " + conversation_id.split(":")[0][:12]) if conversation_id else "CREATE-NEW"
                f.write(f"    -> upstream turn: {act} mode={prompt_mode} tools_in_prompt={bool(tools)} files={len(ref_file_ids)}\n")
        except Exception:
            pass
        return _run_turn(client, prompt, conversation_id, model_type,
                         eff_thinking, req.search, tool_names, req.model,
                         tools_def=tools, ref_file_ids=ref_file_ids)

    # ---- non-streaming ------------------------------------------------------
    if not req.stream:
        def complete_and_respond():
            try:
                turn = serve(prompt, conversation_id, model_type, prompt_mode)
                content, reasoning = [], []
                tool_calls = []
                for kind, v in turn.events():
                    if kind == "content":
                        content.append(v)
                    elif kind == "reasoning":
                        reasoning.append(v)
                    elif kind == "tool_calls":
                        tool_calls = v
                    elif kind == "error":
                        return _error(str(v), status=502, err_type="tool_parse_error")
                text = "".join(content)
                cid = turn.conversation_id
                had_tools = bool(tool_calls)
                echo = [{"role": "assistant",
                         "content": None if tool_calls else text,
                         "tool_calls": [tc.to_openai(i) for i, tc in enumerate(tool_calls)] if tool_calls else None}]
                if title_req:
                    # delete the throwaway conversation upstream (best effort)
                    sid = cid.split(":")[0] if cid else None
                    if sid:
                        try:
                            client._http.post("/api/v0/chat_session/delete",
                                              json={"chat_session_id": sid})
                        except Exception:
                            pass
                elif cid:
                    cmap.remember_session(oc_session, req.model, cid, ref_file_ids=ref_file_ids)
                    if not oc_session:
                        # fingerprint-based clients: store the message shape
                        cmap.remember(req.model + ":" + cid, msg_dicts, cid,
                                      req.model, assistant_echo=echo)
                    rec = cmap._sessions.get(
                        ("ocs:" + oc_session + ":" + req.model) if oc_session
                        else (req.model + ":" + cid))
                    if rec:
                        rec["last_had_tool_calls"] = had_tools
                        cmap._save()
                return completion_response(
                    req.model, text, prompt, cid,
                    reasoning="".join(reasoning),
                    tool_calls=tool_calls or None,
                )
            except RuntimeError as e:
                msg = str(e)
                if _looks_like_invalid_conversation(msg) and conversation_id:
                    cmap.invalidate(conversation_id)
                    return None  # signal caller: retry with fresh conversation
                raise

        try:
            result = await run_in_threadpool(complete_and_respond)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except RuntimeError as e:
            msg = str(e)
            if "401" in msg or "403" in msg or "token" in msg.lower() \
                    or "unauthorized" in msg.lower() or "waf" in msg.lower():
                return _error(
                    "DeepSeek rejected the session (auth/WAF). Run "
                    "`python -m deepseek.auth` in ~/deepseek-api to re-login, "
                    f"then retry. Detail: {msg}",
                    status=503, err_type="login_required",
                )
            return _error(f"DeepSeek request failed: {e}")
        except Exception as e:
            if "timed out" in str(e).lower() or "connect" in str(e).lower():
                return _error(
                    f"Could not reach chat.deepseek.com (network switch or no "
                    f"internet). Check connectivity and retry. Detail: {type(e).__name__}",
                    status=502, err_type="network_error",
                )
            return _error(f"DeepSeek request failed: {e}")

        if result is None:
            # stored conversation was invalid -> transparent recovery with a
            # fresh conversation + full history
            fresh_prompt = build_first_prompt(req.messages, tools)
            def recover():
                turn = _run_turn(client, fresh_prompt, None,
                                 resolve_model_type(req.model),
                                 eff_thinking, req.search, tool_names, req.model,
                                 tools_def=tools, ref_file_ids=ref_file_ids)
                content, reasoning, tool_calls = [], [], []
                for kind, v in turn.events():
                    if kind == "content":
                        content.append(v)
                    elif kind == "reasoning":
                        reasoning.append(v)
                    elif kind == "tool_calls":
                        tool_calls = v
                    elif kind == "error":
                        return _error(str(v), status=502, err_type="tool_parse_error")
                text = "".join(content)
                echo = [{"role": "assistant",
                         "content": None if tool_calls else text,
                         "tool_calls": [tc.to_openai(i) for i, tc in enumerate(tool_calls)] if tool_calls else None}]
                if turn.conversation_id:
                    cmap.remember_session(oc_session, req.model, turn.conversation_id, ref_file_ids=ref_file_ids)
                    if not oc_session:
                        cmap.remember(req.model + ":" + turn.conversation_id,
                                      msg_dicts, turn.conversation_id, req.model,
                                      assistant_echo=echo)
                return completion_response(
                    req.model, text, fresh_prompt, turn.conversation_id,
                    reasoning="".join(reasoning), tool_calls=tool_calls or None,
                )
            try:
                print("[sessions] invalid conversation detected — recovering "
                      "with a fresh DeepSeek conversation (full history)")
                return await run_in_threadpool(recover)
            except Exception as e:
                return _error(f"DeepSeek request failed: {e}")
        return result

    # ---- streaming -----------------------------------------------------------
    # Wrap the generator so mapping bookkeeping runs after consumption, with
    # the full accumulated assistant text available for the echo.
    def gen_with_memory():
        acc_content: list[str] = []
        acc_reasoning: list[str] = []
        tool_calls: list[ToolCall] = []
        finish = "stop"
        try:
            turn = serve(prompt, conversation_id, model_type, prompt_mode)
            for kind, v in turn.events():
                if kind == "content":
                    acc_content.append(v)
                    yield ("content", v)
                elif kind == "reasoning":
                    acc_reasoning.append(v)
                    yield ("reasoning", v)
                elif kind == "tool_calls":
                    tool_calls.extend(v)
                    finish = "tool_calls"
                    yield ("tool_calls", v)
                elif kind == "error":
                    yield ("error", v)
                    return
            cid = turn.conversation_id
            finish = turn.finish_reason
            yield ("done", {"cid": cid, "content": "".join(acc_content),
                            "tool_calls": tool_calls})
        except Exception as e:
            msg = str(e)
            if _looks_like_invalid_conversation(msg) and conversation_id:
                print(f"[sessions] invalid conversation detected during stream: {msg} — invalidating map", flush=True)
                cmap.invalidate(conversation_id)
            yield ("done", {"error": str(e)})

    class _EventAdapter:
        """Adapts (kind, value) tuples into the stream_chunks interface."""

        def __init__(self, src):
            self._src = src
            self.conversation_id = None
            self.finish_reason = "stop"

        def __iter__(self):
            for kind, v in self._src:
                if kind == "error":
                    yield ("error", v)
                    return
                if kind == "done":
                    if "error" in v:
                        raise RuntimeError(v["error"])
                    self.conversation_id = v.get("cid")
                    self.finish_reason = "tool_calls" if v.get("tool_calls") else "stop"
                    # bookkeeping AFTER the stream succeeded
                    if self.conversation_id:
                        had_tools = bool(v.get("tool_calls"))
                        text = v.get("content") or ""
                        echo = [{"role": "assistant",
                                 "content": None if v.get("tool_calls") else text,
                                 "tool_calls": [tc.to_openai(i) for i, tc in enumerate(v["tool_calls"])] if v.get("tool_calls") else None}]
                        cmap.remember_session(oc_session, req.model, self.conversation_id, ref_file_ids=ref_file_ids)
                        if not oc_session:
                            cmap.remember(req.model + ":" + self.conversation_id,
                                          msg_dicts, self.conversation_id, req.model,
                                          assistant_echo=echo)
                        # persist tool-call flag for the next-turn prompt
                        rec = cmap._sessions.get(
                            ("ocs:" + oc_session + ":" + req.model) if oc_session
                            else (req.model + ":" + self.conversation_id))
                        if rec:
                            rec["last_had_tool_calls"] = had_tools
                            cmap._save()
                    continue
                yield (kind, v)

    def sse_stream():
        adapter = _EventAdapter(gen_with_memory())
        try:
            yield from stream_chunks(req.model, adapter)
        except Exception as e:
            err = {"error": {"message": f"DeepSeek stream failed: {e}",
                              "type": "upstream_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
