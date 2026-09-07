"""Passive login watcher: polls the DeepSeek tab in Chrome for the session
token without navigating anywhere. Run in background; when the user completes
sign-in (Google OAuth + any human-check) in Chrome, this captures the session
to session/session.json and exits.

    PYTHONPATH=. .venv/bin/python tools/watch_login.py [--timeout 1800]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepseek.auth import Session  # noqa: E402  (re-binds to android impl)
from deepseek import auth_android as A  # noqa: E402

TOKEN_JS = """
(() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
})()
"""


def main() -> int:
    timeout = 1800.0
    if "--timeout" in sys.argv:
        timeout = float(sys.argv[sys.argv.index("--timeout") + 1])
    deadline = time.time() + timeout
    last_note = 0.0
    print("[watch] Waiting for you to finish signing in to DeepSeek in Chrome...", flush=True)
    print("[watch] (pick YOUR account in the Chrome tab, complete any human-check)", flush=True)

    while time.time() < deadline:
        page = None
        try:
            page, _ = A._open_deepseek_page()
            url = None
            try:
                url = page.evaluate("location.href")
            except Exception:
                pass
            # If we're stuck on a Google OAuth screen or the sign-in page,
            # nudge the tab to chat.deepseek.com so the OAuth/redirect can
            # complete or restart cleanly. Only when no token yet.
            token = page.evaluate(TOKEN_JS)
            if not token and url and (
                "accounts.google.com" in url or url.rstrip("/") == "https://chat.deepseek.com/sign_in"
            ):
                try:
                    page.navigate(A.CHAT_URL)
                except Exception:
                    pass
                time.sleep(2)
                token = page.evaluate(TOKEN_JS)
            if token:
                cookies = {}
                try:
                    res = page.call("Storage.getCookies", timeout=15)
                    for c in res.get("cookies", []):
                        if "deepseek" in (c.get("domain") or ""):
                            cookies[c["name"]] = c["value"]
                except Exception:
                    pass
                ua = page.evaluate("navigator.userAgent") or ""
                s = Session(token=token, cookies=cookies, user_agent=ua,
                            captured_at=time.time())
                s.save()
                print(f"[watch] SESSION CAPTURED (token len {len(token)}, "
                      f"{len(cookies)} cookies) -> session/session.json", flush=True)
                return 0
        except Exception as e:
            if time.time() - last_note > 60:
                print(f"[watch] (waiting; adb/chrome state: {type(e).__name__})", flush=True)
                last_note = time.time()
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
        time.sleep(3)
    print("[watch] timed out without seeing a token", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
