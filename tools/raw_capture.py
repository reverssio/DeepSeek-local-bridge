"""Raw upstream SSE capture for protocol forensics (temporary diagnostic tool).

Talks to chat.deepseek.com directly using the saved session, captures the RAW
SSE frames (unmodified), and writes them redacted to ~/deepseek-api/logs/caps/.

No tokens/cookies are ever written to captures: only the message payload path.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.auth import get_session  # noqa: E402
from deepseek.client import DeepSeekClient  # noqa: E402
from deepseek.pow import DeepSeekPow  # noqa: E402

CAPS = Path(__file__).resolve().parent.parent / "logs" / "caps"


def capture(name: str, prompt: str, *, model: str = "default",
            thinking: bool = False, search: bool = False,
            conversation_id: str | None = None) -> None:
    CAPS.mkdir(parents=True, exist_ok=True)
    sess = get_session(allow_interactive=False)
    client = DeepSeekClient.__new__(DeepSeekClient)
    # minimal init (bypass get_session double-call)
    client.session = sess
    client._pow = DeepSeekPow()
    import threading, httpx
    client._pow_lock = threading.Lock()
    client._http = httpx.Client(
        base_url="https://chat.deepseek.com",
        headers=client._base_headers(),
        cookies=sess.cookies,
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0),
        transport=httpx.HTTPTransport(retries=2),
    )

    from deepseek.client import _biz, _decode_cid, _encode_cid, DEFAULT_MODEL_TYPE, COMPLETION_PATH
    sid, parent = _decode_cid(conversation_id)
    model_type = None
    if sid is None:
        r = client._http.post("/api/v0/chat_session/create", json={})
        sid = _biz(r.json())["chat_session"]["id"]
        model_type = model or DEFAULT_MODEL_TYPE

    body = {
        "chat_session_id": sid,
        "parent_message_id": parent,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "action": None,
        "preempt": False,
    }
    if model_type is not None:
        body["model_type"] = model_type

    pow_solver = DeepSeekPow()
    r = client._http.post(
        "/api/v0/chat/create_pow_challenge", json={"target_path": COMPLETION_PATH}
    )
    challenge = _biz(r.json())["challenge"]
    headers = {"x-ds-pow-response": pow_solver.make_header(challenge)}

    out = open(CAPS / f"{name}.sse.txt", "w", encoding="utf-8")
    meta = {"chat_session_id": sid, "request_body_keys": list(body.keys()),
            "thinking": thinking, "model_type_sent": model_type}
    out.write(f"# meta {json.dumps(meta)}\n")

    with client._http.stream("POST", COMPLETION_PATH, json=body, headers=headers) as resp:
        for line in resp.iter_lines():
            out.write(line + "\n")
    out.close()
    print(f"[cap] {name}: captured -> logs/caps/{name}.sse.txt")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("prompt")
    p.add_argument("--model", default="default")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--cid", default=None)
    a = p.parse_args()
    capture(a.name, a.prompt, model=a.model, thinking=a.thinking,
            search=a.search, conversation_id=a.cid)
