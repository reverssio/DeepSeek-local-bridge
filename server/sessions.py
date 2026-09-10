"""OpenCode-session -> DeepSeek-conversation mapping.

PROBLEM (upstream issue #11 + verified locally): OpenCode never sends the
bridge's custom `conversation_id` field, so every POST created a brand-new
DeepSeek web conversation and re-uploaded the whole flattened backlog. This
pollutes chat.deepseek.com with N conversations per OpenCode session and
wastes upstream context.

DESIGN: a bridge-side session manager that recognizes "this request is a
continuation of that previous request" from the OpenAI message list itself,
and continues the SAME native DeepSeek conversation, sending only the NEW
information (tool results / new user turn) instead of the backlog.

Fingerprint protocol:

  Every request's messages are normalized to a list of (role, content,
  tool_calls_signature, tool_call_id, name) tuples. The bridge stores, for
  each mapped session:
    - the exact message list it served last time (normalized),
    - the assistant reply it returned (text / tool_calls / reasoning),
    - the DeepSeek conversation_id (+ model used to create it).

  A new request is a CONTINUATION iff its normalized messages == the stored
  messages + [the assistant echo(es) we returned] + [new trailing user/tool
  messages]. OpenCode appends exactly what we returned (assistant content, or
  assistant tool_calls + tool results), so this prefix match is deterministic.
  Anything that doesn't match (new OpenCode session, resumed-with-changes,
  bridge restart with lost map) creates a NEW DeepSeek conversation and sends
  the full flattened history — the same behaviour as before, so correctness
  never depends on the map.

Model policy: a DeepSeek conversation's model is fixed at creation. If a
continuation request names a DIFFERENT model than the stored conversation
was created with, we start a NEW conversation (documented behaviour, logged).

Recovery: upstream errors naming an invalid/expired conversation (or a
`parent_message_id` rejection) invalidate the stored mapping; the request
then transparently re-runs against a fresh conversation with the full
history, and the map is re-pointed. Auth refresh doesn't invalidate
conversations (they live server-side).

Title/summarization requests (OpenCode sends a small "generate a title"
system prompt alongside real ones) are detected by signature and routed to
their own short-lived conversation which is deleted upstream afterwards —
they must not pollute or reuse the main conversation.

Thread-safety: all map operations take a lock; the upstream request itself
is serialized by the server's single-client lock, so races are impossible.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

# Signature of OpenCode's title/summarization system prompt.
_TITLE_MARKERS = (
    "generate a title for the conversation",
    "never include tool names in the title",
    "you generate a title",
)
# Title prompts are also TINY compared to the main agent prompt.
_TITLE_MAX_SYS_LEN = 3500

MAP_PATH = Path(__file__).resolve().parent.parent / "session" / "conv_map.json"
MAX_MAPPED_SESSIONS = 64  # FIFO eviction; map holds no secrets


def is_title_request(messages: List[dict]) -> bool:
    """Detect OpenCode's background title/summarization requests."""
    for m in messages:
        if m.get("role") != "system":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        low = c.lower()
        if any(marker in low for marker in _TITLE_MARKERS) and len(c) <= _TITLE_MAX_SYS_LEN:
            return True
    return False


def _norm_messages(messages: List[dict]) -> List[dict]:
    """Normalize OpenAI messages into comparable dicts (drops reasoning).

    Assistant tool-call turns are compared STRUCTURALLY: clients echo them
    back with content=null (OpenCode) or with the parsed text, so assistant
    content is ignored when tool_calls are present — the tool-call signature
    (names + arguments) is the identity.
    """
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):  # vision-style parts: keep text only
            c = "\n".join(p.get("text", "") for p in c
                          if isinstance(p, dict) and p.get("type") == "text")
        tcs = m.get("tool_calls") or []
        tc_sig = [(t.get("function", {}).get("name"),
                   t.get("function", {}).get("arguments"))
                  for t in tcs if isinstance(t, dict)]
        if m.get("role") == "assistant" and tc_sig:
            content_key = None  # identity is the tool-call signature
        else:
            content_key = c if isinstance(c, (str, type(None))) else str(c)
        out.append({
            "role": m.get("role"),
            "content": content_key,
            "tc": tc_sig,
            "tcid": m.get("tool_call_id"),
            "name": m.get("name"),
        })
    return out


def _match_prefix(stored: List[dict], incoming: List[dict]) -> bool:
    """True when `stored` is a prefix of `incoming` (exact tuple equality)."""
    if len(stored) > len(incoming):
        return False
    for a, b in zip(stored, incoming):
        if a != b:
            return False
    return True


class ConversationMap:
    """Persistent mapping of bridge-served sessions to DeepSeek conversations."""

    def __init__(self, path: Path = MAP_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._sessions = data
        except Exception:
            self._sessions = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # FIFO eviction, so the file stays small
            if len(self._sessions) > MAX_MAPPED_SESSIONS:
                for k in sorted(
                    self._sessions,
                    key=lambda k: self._sessions[k].get("last_used", 0),
                )[: len(self._sessions) - MAX_MAPPED_SESSIONS]:
                    del self._sessions[k]
            self._path.write_text(json.dumps(self._sessions))
        except Exception:
            pass  # map is an optimization; never fail a request over it

    # -- matching -----------------------------------------------------------

    def find_by_session(self, oc_session: str, model: str) -> Optional[dict]:
        """Direct lookup by OpenCode session id (header) + model.

        This is the PRIMARY mapping path when OpenCode sends its session id
        (x-session-affinity / X-Session-Id, verified present in 1.18.27
        traffic). One OpenCode session + model -> exactly one DeepSeek
        conversation, regardless of message shape."""
        if not oc_session:
            return None
        key = f"ocs:{oc_session}:{model}"
        with self._lock:
            rec = self._sessions.get(key)
            if rec:
                rec["last_used"] = time.time()
                self._save()
            return rec

    def find(self, messages: List[dict]) -> Optional[dict]:
        """Return the session record this request continues, if any.

        Fallback path when no session header is available: match by message
        fingerprint (stored request prefix + assistant echo)."""
        norm = _norm_messages(messages)
        with self._lock:
            for key, rec in self._sessions.items():
                stored_norm = rec.get("messages_norm")
                if not stored_norm:
                    continue  # session-keyed records don't fingerprint-match
                # Match: stored request prefix + our returned assistant
                # message(s) appended, then new trailing messages allowed.
                assistant_echo = rec.get("assistant_echo_norm") or []
                full_stored = stored_norm + assistant_echo
                if _match_prefix(full_stored, norm):
                    return rec
        return None

    def remember_session(self, oc_session: str, model: str,
                         conversation_id: str,
                         ref_file_ids: Optional[list[str]] = None) -> None:
        """Remember the DeepSeek conversation for an OpenCode session+model."""
        if not oc_session or not conversation_id:
            return
        key = f"ocs:{oc_session}:{model}"
        session_id, _, msg_part = conversation_id.partition(":")
        msg_id = int(msg_part) if msg_part.isdigit() else None
        with self._lock:
            rec = self._sessions.get(key) or {
                "created_at": time.time(),
            }
            existing_files = list(rec.get("ref_file_ids") or [])
            if ref_file_ids:
                for fid in ref_file_ids:
                    if fid not in existing_files:
                        existing_files.append(fid)
            rec.update({
                "conversation_id": conversation_id,
                "chat_session_id": session_id,
                "last_message_id": msg_id,
                "model": model,
                "ref_file_ids": existing_files,
                "has_vision": bool(existing_files),
                "last_used": time.time(),
            })
            self._sessions[key] = rec
            self._save()

    def remember(self, key: str, messages: List[dict], conversation_id: str,
                 model: str, assistant_echo: Optional[List[dict]] = None,
                 title: bool = False) -> None:
        """Store/refresh the mapping after serving a request.

        `assistant_echo` is what the bridge RETURNED for this request (the
        message OpenCode will append to its history)."""
        rec = {
            "messages_norm": _norm_messages(messages),
            "assistant_echo_norm": _norm_messages(assistant_echo or []),
            "conversation_id": conversation_id,
            "model": model,
            "title": title,
            "created_at": time.time(),
            "last_used": time.time(),
        }
        with self._lock:
            self._sessions[key] = rec
            self._save()

    def touch(self, key: str) -> None:
        with self._lock:
            rec = self._sessions.get(key)
            if rec:
                rec["last_used"] = time.time()
                self._save()

    def invalidate(self, conversation_id: str) -> None:
        """Drop every mapping pointing at an invalid conversation."""
        with self._lock:
            dead = [k for k, r in self._sessions.items()
                    if r.get("conversation_id") == conversation_id]
            for k in dead:
                del self._sessions[k]
            self._save()

    def stats(self) -> dict:
        with self._lock:
            return {
                "mapped_sessions": len(self._sessions),
                "conversations": sorted({
                    r["conversation_id"].split(":")[0]
                    for r in self._sessions.values()
                }),
            }


# module-level singleton used by the server
_map: Optional[ConversationMap] = None
_map_lock = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()
_fallback_session_lock = threading.Lock()


def get_session_lock(oc_session: Optional[str]) -> threading.Lock:
    """Return a concurrency-safe single-flight lock per OpenCode session."""
    if not oc_session:
        return _fallback_session_lock
    with _session_locks_guard:
        if oc_session not in _session_locks:
            _session_locks[oc_session] = threading.Lock()
        return _session_locks[oc_session]


def get_map() -> ConversationMap:
    global _map
    with _map_lock:
        if _map is None:
            _map = ConversationMap()
        return _map
