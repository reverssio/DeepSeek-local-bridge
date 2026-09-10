"""
Android/Termux authentication — CDP over ADB against the phone's real Chrome.

This is a drop-in replacement for the upstream Playwright-based auth on
platforms where Playwright's bundled Chromium cannot run (Android/Termux,
non-rooted). It uses the SAME legitimate mechanism Playwright itself uses for
Android automation (https://playwright.dev/docs/android): Chrome's devtools
socket exposed over ADB:

    adb forward tcp:PORT localabstract:chrome_devtools_remote_<pid>

We then speak the Chrome DevTools Protocol directly over WebSocket:
  - Page.navigate to chat.deepseek.com / sign_in
  - Runtime.evaluate to read localStorage.userToken (same JS as upstream)
  - Network.getAllCookies / Storage.getCookies to capture session cookies
  - Runtime.evaluate to read navigator.userAgent

The user signs in BY HAND in the visible Chrome window (their own account,
their own fingers for the WAF/human check). We only observe the session that
their own browser holds, exactly like the upstream code does with Playwright.

No CAPTCHA/WAF bypass: the human check is completed by the human, on their own
device, in their own browser.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import httpx
from websockets.sync.client import connect as ws_connect

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_DIR = Path(os.getenv("DEEPSEEK_PROFILE_DIR", ROOT / "session" / "profile"))
DEFAULT_SESSION_FILE = ROOT / "session" / "session.json"

CHAT_URL = "https://chat.deepseek.com/"
SIGNIN_URL = "https://chat.deepseek.com/sign_in"

SESSION_MAX_AGE = 6 * 60 * 60  # 6 hours, same as upstream

# Local TCP port used for the adb forward to Chrome's devtools socket.
CDP_PORT = int(os.getenv("DEEPSEEK_CDP_PORT", "17025"))
ADB_TARGET = os.getenv("DEEPSEEK_ADB_TARGET", "")  # optional adb -s target

_READ_TOKEN_JS = """
(() => {
  try {
    const raw = window.localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
})()
"""

_READ_UA_JS = "navigator.userAgent"


class LoginRequired(RuntimeError):
    """No usable session and interactive login is disallowed."""

    DEFAULT = (
        "No DeepSeek session found. Log in first by running:\n"
        "    python -m deepseek.auth\n"
        "This opens Chrome (via ADB/CDP) so you can sign in and clear the "
        "human-check; afterwards the server reuses the saved session automatically."
    )

    def __init__(self, message: str = DEFAULT):
        super().__init__(message)


@dataclass
class Session:
    """A captured signed-in DeepSeek session (same shape as upstream)."""

    token: str
    cookies: Dict[str, str]
    user_agent: str
    captured_at: float

    @property
    def age(self) -> float:
        return time.time() - self.captured_at

    def save(self, path: Path = DEFAULT_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_SESSION_FILE) -> Optional["Session"]:
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


# --- ADB / CDP plumbing ------------------------------------------------------


def _adb(*args: str, _retried: bool = False) -> str:
    cmd = ["adb"]
    if ADB_TARGET:
        cmd += ["-s", ADB_TARGET]
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        # Wireless adb drops occasionally; one reconnect attempt (only for
        # shell queries, and only once, to avoid loops).
        if not _retried and "no devices" in err.lower():
            _adb_reconnect()
            return _adb(*args, _retried=True)
        raise RuntimeError(f"adb {' '.join(args)} failed: {err}")
    return r.stdout.strip()


def _adb_reconnect() -> None:
    """Best-effort wireless-adb reconnect (adb connect to the known target)."""
    try:
        target = ADB_TARGET or _last_known_adb_target()
        if target:
            subprocess.run(
                ["adb", "connect", target], capture_output=True, text=True, timeout=15
            )
    except Exception:
        pass


# File remembering the last adb target that worked (host:port of the phone's
# wireless-debugging adbd). The port rotates when wireless debugging restarts,
# but remembering it lets us try the previous combination before giving up.
_ADB_TARGET_FILE = ROOT / "session" / "adb_target.txt"


def _remember_adb_target(target: str) -> None:
    try:
        _ADB_TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ADB_TARGET_FILE.write_text(target + "\n")
    except Exception:
        pass


def _saved_adb_target() -> Optional[str]:
    try:
        return _ADB_TARGET_FILE.read_text().strip() or None
    except Exception:
        return None


def _last_known_adb_target() -> Optional[str]:
    """Find a connected adb device target, else the last one that worked.

    We do not guess host:port combos — Android's wireless-debugging port
    rotates and wrong guesses are noise. Preference order: a currently
    attached device (from `adb devices`), then the remembered target file.
    """
    try:
        r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device" and ":" in parts[0]:
                _remember_adb_target(parts[0])
                return parts[0]
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device":
                _remember_adb_target(parts[0])
                return parts[0]
    except Exception:
        pass
    return _saved_adb_target()


def _find_devtools_socket() -> str:
    """Find the chrome devtools abstract socket name from /proc/net/unix."""
    try:
        out = _adb("shell", "cat /proc/net/unix")
    except RuntimeError as e:
        raise RuntimeError(
            "ADB is not connected, so Chrome's devtools socket is unreachable.\n"
            "On the phone: Settings -> Developer options -> Wireless debugging ->\n"
            "use 'Pair device with pairing code' once, then in Termux run:\n"
            "    adb pair <ip:pairing-port> <code>\n"
            "    adb connect <ip:port>\n"
            f"(previous target was: {_saved_adb_target() or 'unknown'})\n"
            f"Underlying error: {e}"
        ) from e
    for line in out.splitlines():
        if "chrome_devtools_remote" in line:
            # last field is the socket name, e.g. @chrome_devtools_remote_17025
            name = line.split()[-1]
            if name.startswith("@"):
                name = name[1:]  # adb forward wants the name without '@'
            return name
    raise RuntimeError(
        "No chrome_devtools_remote socket found. Open Chrome on the phone "
        "(or enable it via Settings). Chrome must be running."
    )


def _ensure_forward() -> None:
    """Forward local CDP_PORT to Chrome's devtools socket (idempotent-ish)."""
    name = _find_devtools_socket()
    _adb("forward", f"tcp:{CDP_PORT}", f"localabstract:{name}")


def _http_targets() -> list:
    r = httpx.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=10)
    r.raise_for_status()
    return r.json()


class _CDPPage:
    """Minimal CDP page client over websockets."""

    def __init__(self, ws_url: str):
        self._ws = None
        self._id = 0
        self._ws_url = ws_url

    # -- connection ----------------------------------------------------------

    def connect(self, timeout: float = 20.0):
        self._ws = ws_connect(
            self._ws_url, max_size=None, open_timeout=timeout
        )

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _recv_until(self, mid: int, timeout: float = 30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = self._ws.recv(timeout=max(0.5, deadline - time.time()))
            except TimeoutError:
                break
            msg = json.loads(raw)
            if msg.get("id") == mid:
                return msg
        raise TimeoutError(f"CDP: no response for message id {mid}")

    def call(self, method: str, params: dict = None, timeout: float = 30.0):
        self._id += 1
        mid = self._id
        self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        resp = self._recv_until(mid, timeout=timeout)
        if "error" in resp:
            raise RuntimeError(f"CDP {method} error: {resp['error']}")
        return resp.get("result", {})

    # -- helpers --------------------------------------------------------------

    def navigate(self, url: str, timeout: float = 45.0) -> None:
        """Navigate; tolerate interrupted loads (SPA redirects, WAF checks)."""
        try:
            self.call("Page.navigate", {"url": url}, timeout=timeout)
        except TimeoutError:
            print(f"[auth] navigation to {url} interrupted; continuing")
        # give the SPA a moment to render
        time.sleep(2.0)

    def evaluate(self, expression: str, await_promise: bool = False, timeout: float = 20.0):
        """Evaluate JS in the page; return (result, exceptionText)."""
        params = {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
        }
        try:
            res = self.call("Runtime.evaluate", params, timeout=timeout)
        except TimeoutError:
            return None
        if res.get("exceptionDetails"):
            return None
        remote = res.get("result", {})
        if remote.get("subtype") == "null":
            return None
        return remote.get("value")


def _open_deepseek_page() -> tuple[_CDPPage, list]:
    """Ensure the adb forward, find or open a DeepSeek tab, return (page, targets)."""
    _ensure_forward()
    targets = _http_targets()
    # Look for an existing chat.deepseek.com page; else open a new tab.
    ds = [t for t in targets if t.get("type") == "page" and "deepseek" in (t.get("url") or "")]
    if ds:
        target = ds[0]
    else:
        # /json/new is PUT in newer Chrome versions
        try:
            r = httpx.put(
                f"http://127.0.0.1:{CDP_PORT}/json/new?{CHAT_URL}",
                timeout=15,
            )
            target = r.json()
        except Exception:
            # fall back to reusing the first ordinary page tab
            pages = [t for t in targets if t.get("type") == "page"]
            if not pages:
                raise RuntimeError("No browser page available via CDP")
            target = pages[0]
    page = _CDPPage(target["webSocketDebuggerUrl"])
    page.connect()
    return page, targets


def _capture_session(page: _CDPPage) -> Optional[Session]:
    token = page.evaluate(_READ_TOKEN_JS)
    if not token:
        return None
    cookies: Dict[str, str] = {}
    try:
        res = page.call("Storage.getCookies", timeout=15)
        for c in res.get("cookies", []):
            if "deepseek" in (c.get("domain") or ""):
                cookies[c["name"]] = c["value"]
    except Exception:
        pass
    if not cookies:  # older Chrome: Network.getAllCookies
        try:
            res = page.call("Network.getAllCookies", timeout=15)
            for c in res.get("cookies", []):
                if "deepseek" in (c.get("domain") or ""):
                    cookies[c["name"]] = c["value"]
        except Exception:
            pass
    ua = page.evaluate(_READ_UA_JS) or ""
    return Session(token=token, cookies=cookies, user_agent=ua, captured_at=time.time())


def _wait_for_token(page: _CDPPage, timeout: float = 300) -> Optional[str]:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        token = page.evaluate(_READ_TOKEN_JS)
        if token:
            return token
        # Show the user a hint about where we are in the login flow.
        url = None
        try:
            url = page.evaluate("location.href")
        except Exception:
            pass
        if url != last_status:
            print(f"[auth] waiting for sign-in... (page: {url})")
            last_status = url
        time.sleep(2.0)
    return None


# --- public API (mirrors upstream deepseek/auth.py) --------------------------


def login(
    profile_dir: Path = DEFAULT_PROFILE_DIR,  # unused on Android (Chrome owns its profile)
    headless: bool = False,                    # unused — Chrome is the real, visible browser
    assume_logged_out: bool = False,
) -> Session:
    """Wait for the user to sign in to chat.deepseek.com in Chrome, then capture.

    The user signs in with their own account, in their own Chrome, on their own
    phone. We just poll localStorage.userToken through CDP until it appears.
    The AWS WAF human-check is completed by the human, by hand, in Chrome.
    """
    page, _ = _open_deepseek_page()
    try:
        page.navigate(CHAT_URL)
        token = page.evaluate(_READ_TOKEN_JS)
        if not token:
            print(
                "[auth] Opening the DeepSeek sign-in page in Chrome. Please log in "
                "with YOUR account in the Chrome window (solve the human-check if "
                "shown). Waiting for the session..."
            )
            page.navigate(SIGNIN_URL)
            if not _wait_for_token(page, timeout=300):
                raise RuntimeError("Login timed out — no token captured.")
        session = _capture_session(page)
        if session is None:
            raise RuntimeError("Logged in but could not read the token.")
        session.save()
        return session
    finally:
        page.close()


def _headless_refresh(profile_dir: Path = None) -> Optional[Session]:
    """Try to capture a token from the already-signed-in Chrome (no UI action).

    On Android the 'profile' is the real Chrome profile — if the user is still
    signed in there, this captures a fresh token without any interaction.
    """
    try:
        page, _ = _open_deepseek_page()
    except Exception as e:
        print(f"[auth] headless refresh unavailable: {e}")
        return None
    try:
        page.navigate(CHAT_URL)
        time.sleep(3.0)
        session = _capture_session(page)
    except Exception as e:
        print(f"[auth] headless refresh failed: {e}")
        session = None
    finally:
        page.close()
    if session is not None:
        session.save()
    return session


def get_session(
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    session_file: Path = DEFAULT_SESSION_FILE,
    max_age: int = SESSION_MAX_AGE,
    allow_interactive: bool = True,
) -> Session:
    """Return a usable session.

    Policy (Android/Termux): OPTIMISTIC token reuse. The cached session file
    is used as long as it exists — DeepSeek bearer tokens routinely outlive
    our 6h heuristic (verified: token still valid 7.6h in), and attempting a
    Chrome/CDP refresh on every stale-but-valid token blocks requests whenever
    adb is disconnected. Real expiry is detected by the server (401/403 from
    upstream), which then reports `login_required` to the client; only the
    explicit `python -m deepseek.auth` command performs the interactive
    Chrome flow.

    If NO session file exists at all and interactive login is allowed, a
    login is attempted (this only happens on first ever use).
    """
    cached = Session.load(session_file)
    if cached is not None:
        return cached

    if not allow_interactive:
        raise LoginRequired()

    # No session at all — first ever use: try a headless capture (maybe Chrome
    # is already signed in), then fall back to interactive login.
    session = _headless_refresh(profile_dir)
    if session is not None:
        return session

    print("[auth] No valid session found — Chrome will open the sign-in page...")
    return login(assume_logged_out=True)


if __name__ == "__main__":
    s = login()
    print(f"[auth] captured token {s.token[:10]}... ({len(s.cookies)} cookies)")
