"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow
from .sse import parse_sse_events

_pow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="deepseek-pow")

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    # DeepThink reasoning, kept SEPARATE from the visible answer text
    # ("" when thinking was disabled for the request).
    reasoning: str = ""

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._auth_lock = threading.Lock()
        # Android note: when the phone switches Wi-Fi <-> mobile data, already
        # established (pooled, keep-alive) TCP sockets to chat.deepseek.com die
        # silently -- no FIN/RST arrives, so a request sent on one stalls until
        # the read timeout. We therefore:
        #   * keep connections in the pool only briefly (keepalive_expiry),
        #   * retry a bounded number of times on transport errors --
        #     httpx's transport retries are safe here: they only retry
        #     requests whose body was fully sent (POSTs below are JSON with
        #     idempotent server-side effects, and a replayed request the
        #     server already processed fails cleanly at the HTTP layer
        #     rather than being silently lost),
        #   * use a read timeout short enough to notice a dead stream but
        #     long enough for normal completion generation (was 300s).
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0),
            limits=httpx.Limits(max_keepalive_connections=2, keepalive_expiry=15.0),
            transport=httpx.HTTPTransport(retries=2),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        sid = _biz(r.json())["chat_session"]["id"]
        # Visible marker so conversation creation is countable in server logs
        print(f"[deepseek] created chat_session {sid}", flush=True)
        return sid

    def get_session_current_message_id(self, chat_session_id: str) -> Optional[int]:
        """Fetch the current head message ID of a chat session from DeepSeek Web."""
        try:
            r = self._http.get(f"/api/v0/chat/history_messages?chat_session_id={chat_session_id}")
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == 0:
                    biz = data.get("data", {}).get("biz_data", {})
                    cur_mid = biz.get("chat_session", {}).get("current_message_id")
                    if isinstance(cur_mid, int):
                        return cur_mid
                    msgs = biz.get("chat_messages", [])
                    if msgs and isinstance(msgs[-1].get("message_id"), int):
                        return msgs[-1].get("message_id")
        except Exception as e:
            print(f"[deepseek] failed to get current_message_id for {chat_session_id}: {e}", flush=True)
        return None

    def refresh_auth(self) -> bool:
        """Attempt single-flight reload or headless refresh of session token."""
        with self._auth_lock:
            try:
                from .auth import get_session
                new_sess = get_session(allow_interactive=False)
                if new_sess and new_sess.token:
                    self.session = new_sess
                    self._http.headers.update(self._base_headers())
                    self._http.cookies.update(self.session.cookies)
                    print("[auth] refreshed session token successfully", flush=True)
                    return True
            except Exception as e:
                print(f"[auth] session refresh failed: {e}", flush=True)
            return False

    def _pow_header(self, target_path: str = COMPLETION_PATH, timeout: float = 15.0) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]

        def _solve():
            with self._pow_lock:
                return self._pow.make_header(challenge)

        future = _pow_executor.submit(_solve)
        return future.result(timeout=timeout)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list[str]] = None,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search, ref_file_ids=ref_file_ids)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list[str]] = None,
    ) -> Reply:
        """Return the complete reply (`.text`, `.reasoning`) plus its
        `.conversation_id`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search,
                        ref_file_ids=ref_file_ids)
        text, reasoning = [], []
        for kind, delta in s.events():
            if kind == "content":
                text.append(delta)
            elif kind == "reasoning":
                reasoning.append(delta)
        return Reply(text="".join(text), conversation_id=s.conversation_id,
                     reasoning="".join(reasoning))

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Streamed reply. Iterate `events()` for typed chunks — ("content", text)
    or ("reasoning", text) — or iterate the object itself for plain text deltas
    (backwards compatible). After consumption, `.conversation_id` holds the
    token for resuming the conversation."""

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: str,
                 thinking: bool, search: bool,
                 ref_file_ids: Optional[list[str]] = None):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._ref_file_ids = ref_file_ids or []
        self._message_id: Optional[int] = parent_id

    def events(self):
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": self._ref_file_ids,
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        meta: dict = {}
        # Android/Wi-Fi-switch note: a pooled socket that died mid-flight
        # (network switched under it) surfaces as a transport error on THIS
        # request. If nothing was yielded yet, one retry with a fresh PoW
        # challenge is safe and covers the common "switch happened between
        # requests / right at request start" case. If bytes were already
        # emitted, retrying would duplicate output, so we raise instead.
        yielded_any = False
        for attempt in range(3):
            # PoW challenges are short-lived, so solve right before the request.
            headers = {"x-ds-pow-response": self._client._pow_header()}
            try:
                with self._client._http.stream(
                    "POST", COMPLETION_PATH, json=body, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if ctype.startswith("application/json"):
                        # DeepSeek returned an application/json business error payload instead of an SSE stream
                        err_data = json.loads(resp.read().decode("utf-8", errors="replace"))
                        biz_code = err_data.get("data", {}).get("biz_code") if isinstance(err_data.get("data"), dict) else None
                        biz_msg = err_data.get("data", {}).get("biz_msg") or err_data.get("msg") or str(err_data)
                        if (biz_code == 26 or "invalid message id" in str(biz_msg).lower()) and attempt < 2 and not yielded_any:
                            print(f"[deepseek] parent_message_id {body.get('parent_message_id')} desynchronized (biz_code 26). Resynchronizing with session head...", flush=True)
                            real_mid = self._client.get_session_current_message_id(self._session_id)
                            if real_mid is not None and real_mid != body.get("parent_message_id"):
                                print(f"[deepseek] resynchronized parent_message_id: {body.get('parent_message_id')} -> {real_mid}", flush=True)
                                body["parent_message_id"] = real_mid
                                self._parent_id = real_mid
                                continue
                            elif real_mid is None and body.get("parent_message_id") is not None:
                                print(f"[deepseek] resynchronized parent_message_id {body.get('parent_message_id')} -> None (session head fallback)", flush=True)
                                body["parent_message_id"] = None
                                self._parent_id = None
                                continue
                        raise RuntimeError(f"DeepSeek upstream business error: {biz_msg} (biz_code={biz_code})")

                    for ev in parse_sse_events(resp.iter_lines(), meta):
                        kind = ev[0]
                        if kind in ("content", "reasoning"):
                            yielded_any = True
                        yield ev
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == 0 and not yielded_any:
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403):
                        self._client.refresh_auth()
                    continue  # single bounded retry with fresh PoW and auth
                if yielded_any:
                    raise RuntimeError(
                        "DeepSeek stream interrupted mid-response (network "
                        "switch or upstream drop). Partial output was "
                        "already delivered; ask again to regenerate."
                    ) from e
                raise
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]

    def __iter__(self) -> Iterator[str]:
        for kind, delta in self.events():
            if kind == "content":
                yield delta

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[str]:
    """Backwards-compatible text-only view over the typed event parser.

    Kept because external code (examples/) iterates client.stream() directly.
    NOTE: this view concatenates reasoning into text only if the caller opts
    in via the `events()` API — no: text-only yields RESPONSE content alone.
    """
    for kind, delta in parse_sse_events(lines, meta):
        if kind == "content":
            yield delta
