"""Fragment-aware parser for DeepSeek's SSE completion protocol.

DeepSeek streams a response object patch protocol. A response contains an
ordered list of `fragments`, each with a `type` — in practice "THINK"
(DeepThink reasoning) and "RESPONSE" (the visible answer). The wire events:

  data: {"v": {"response": {..., "fragments":[{"id":2,"type":"THINK",
           "content":"1", ...}]}}}                       <- snapshot frame
  data: {"p":"response/fragments/-1/content","o":"APPEND","v":" The"}
                                                        <- append to LAST fragment
  data: {"p":"response/fragments","o":"APPEND","v":[{"id":3,
           "type":"RESPONSE","content":"42", ...}]}     <- NEW fragment(s)
  data: {"v":"'s"}                                      <- bare append to the
                                                          current path
  data: {"p":"response/status","o":"SET","v":"FINISHED"}

The previous parser kept only a single "active path" string and yielded every
content append, which (a) concatenated THINK and RESPONSE text, and (b)
ignored the initial content of newly appended fragments (the "eaten first
token" bug, upstream issue #9).

This module yields typed events instead:

    ("reasoning", text)   -- a THINK fragment delta
    ("content", text)     -- a RESPONSE fragment delta
    ("finish", None)      -- response status FINISHED

and records `message_id` into a meta dict (used for resumable
conversation_id; the snapshot frame carries it directly under
response.message_id, which is more reliable than path-watching).
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

# fragment types that carry user-visible answer text vs reasoning
THINK = "THINK"
RESPONSE = "RESPONSE"


def format_search_citations(search_results: list[dict]) -> str:
    """Format DeepSeek search results into Markdown citation references."""
    if not search_results:
        return ""
    valid = [r for r in search_results if isinstance(r, dict) and r.get("cite_index") and r.get("url")]
    valid.sort(key=lambda r: int(r.get("cite_index", 0)))
    lines = [f"[{r['cite_index']}]: [{r.get('title', 'Source')}]({r['url']})" for r in valid]
    return "\n\n" + "\n".join(lines) if lines else ""


class Fragments:
    """Mutable view of the response's fragment list as patches arrive."""

    def __init__(self) -> None:
        self.frags: list[dict] = []

    def apply_snapshot(self, response: dict) -> None:
        frags = response.get("fragments")
        if isinstance(frags, list):
            # The snapshot can arrive multiple times mid-stream (it re-states
            # the whole object). Reset to the authoritative value, but keep
            # any content we already emitted accounted for: the snapshot's
            # fragment content includes everything streamed so far, so
            # replacing wholesale is correct.
            self.frags = [f for f in frags if isinstance(f, dict)]

    def append_fragments(self, new_frags: list) -> None:
        for f in new_frags:
            if isinstance(f, dict):
                self.frags.append(f)

    def append_content(self, text: str) -> Optional[str]:
        """Append text to the last fragment; returns its type (or None)."""
        if not self.frags or not isinstance(text, str):
            return None
        last = self.frags[-1]
        last["content"] = (last.get("content") or "") + text
        return last.get("type")

    def set_path_content(self, text: str) -> Optional[str]:
        """SET on .../content: replace last fragment's content."""
        if not self.frags or not isinstance(text, str):
            return None
        last = self.frags[-1]
        last["content"] = text
        return last.get("type")


def parse_sse_events(lines, meta: Optional[dict] = None) -> Iterator[tuple]:
    """Turn DeepSeek's SSE completion stream into typed events.

    Yields ("reasoning", delta) for THINK text and ("content", delta) for
    RESPONSE text; ("finish", None) once the response status becomes
    FINISHED. `meta`, if given, receives "message_id" and "status".
    """
    frags = Fragments()
    active_path: Optional[str] = None
    # Paths whose patches mean "append to last fragment's content". DeepSeek
    # uses -1 as the index of the last array element.
    CONTENT_PATH_SUFFIXES = (
        "response/fragments/-1/content",
    )
    emitted_first: dict[int, bool] = {}  # fragment id -> initial content emitted

    search_results: list[dict] = []
    if meta is not None:
        meta["search_results"] = search_results

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        v = obj.get("v")
        p = obj.get("p")
        o = obj.get("o")

        # Capture response_message_id from top-level framing if present
        if meta is not None and isinstance(obj.get("response_message_id"), int):
            meta["message_id"] = obj["response_message_id"]

        # --- snapshot frame: full response object -------------------------
        if isinstance(v, dict) and "response" in v:
            resp = v["response"]
            if meta is not None and isinstance(resp.get("message_id"), int):
                meta["message_id"] = resp["message_id"]
            frags.apply_snapshot(resp)
            # Emit any fragment content we haven't emitted yet (covers the
            # initial THINK/RESPONSE text carried by the snapshot itself).
            for f in frags.frags:
                fid = f.get("id")
                if not emitted_first.get(fid) and f.get("content"):
                    emitted_first[fid] = True
                    kind = "reasoning" if f.get("type") == THINK else "content"
                    if f.get("type") in (THINK, RESPONSE):
                        yield (kind, f["content"])
            continue

        # --- patch frames ---------------------------------------------------
        if p is not None:
            active_path = p

            # NEW fragment(s) appended to the fragments array.
            if p == "response/fragments" and o == "APPEND" and isinstance(v, list):
                before = {f.get("id") for f in frags.frags}
                frags.append_fragments(v)
                for f in v:
                    if isinstance(f, dict) and f.get("id") not in before \
                            and f.get("content"):
                        emitted_first[f["id"]] = True
                        kind = "reasoning" if f.get("type") == THINK else "content"
                        if f.get("type") in (THINK, RESPONSE):
                            yield (kind, f["content"])
                continue

            # Append/SET to the last fragment's content.
            if p.endswith("content") and isinstance(v, str):
                if any(p.endswith(suf) for suf in CONTENT_PATH_SUFFIXES):
                    ftype = frags.append_content(v) if o != "SET" \
                        else frags.set_path_content(v)
                else:
                    # e.g. "response/fragments/2/content" (indexed) — apply
                    # to the last fragment as well; DeepSeek has only ever
                    # observed -1 in practice.
                    ftype = frags.append_content(v) if o != "SET" \
                        else frags.set_path_content(v)
                if ftype in (THINK, RESPONSE):
                    yield ("reasoning" if ftype == THINK else "content", v)
                continue

            # Search results.
            if p == "response/search_results" and isinstance(v, list):
                if o != "BATCH":
                    search_results.clear()
                    search_results.extend([x for x in v if isinstance(x, dict)])
                else:
                    for op in v:
                        if isinstance(op, dict) and "p" in op and "v" in op:
                            import re as _re
                            m = _re.match(r"^(\d+)/cite_index$", str(op.get("p", "")))
                            if m:
                                idx = int(m.group(1))
                                if idx < len(search_results):
                                    search_results[idx]["cite_index"] = op["v"]
                continue

            # Status transitions.
            if p == "response/status" and isinstance(v, str):
                if meta is not None:
                    meta["status"] = v
                if v == "FINISHED":
                    citations = format_search_citations(search_results)
                    if citations:
                        yield ("content", citations)
                    yield ("finish", None)
            continue

        # --- bare value frame: append to the current path ---------------------
        if isinstance(v, str) and active_path and active_path.endswith("content"):
            ftype = frags.append_content(v)
            if ftype in (THINK, RESPONSE):
                yield ("reasoning" if ftype == THINK else "content", v)


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Kept for backwards compatibility with the old parser's tests."""
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
