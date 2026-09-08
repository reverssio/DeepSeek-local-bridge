"""Lenient tool-call parser: handles every dialect the DeepSeek web model
emits, including malformed hybrids of the instructed format.

Observed dialects (all from live traffic on this account):
  1. <tool>{"name": "bash", "arguments": {...}}</tool>          (instructed)
  2. <tool><parameter name="command">ls</parameter></tool>       (XML params)
  3. <tool name="bash">{...}</tool>                              (named tag)
  4. <tool><name>bash</name><arguments><command>x</command>
     </arguments></tool>                                          (full XML)
  5. <tool>{"name": "bash">                                       (broken JSON
     <parameter name="command">ls</parameter></tool>               hybrid)
  6. <｜DSML｜tool_calls><｜DSML｜invoke name="Bash">...          (native DSML)
  7. <bash>ls</bash>                                              (bare tags)
  8. truncated closers: '...</tool', '...</invoke', degenerate
     repeated openers

Strategy: find the tool-call REGION (from the first recognized opener to the
last recognized closer, or EOF), then inside it extract tool name from
whichever pattern appears (JSON "name", XML <name>, tag attr, DSML invoke,
bare tag), and arguments from whichever pattern appears (JSON "arguments",
<parameter>/<command> tags, DSML parameters). Anything unparseable is
removed from the returned clean text (never leaked as assistant content).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import List, Optional, Tuple

# --- region delimiters --------------------------------------------------------

_OPENERS = [
    re.compile(r"<tool\b[^>]*>"),
    re.compile(r"<｜DSML｜tool_calls>"),
    re.compile(r"<bash>|<edit>|<read>|<write>|<grep>|<glob>"),
]
_CLOSERS = [
    re.compile(r"</tool\s*>|</tool$|</tool"),
    re.compile(r"</｜DSML｜tool_calls>"),
    re.compile(r"</bash>|</edit>|</read>|</write>|</grep>|</glob>"),
]

# --- name/argument extractors (applied inside a region) -----------------------

_NAME_JSON = re.compile(r'"name"\s*:\s*"([^"]+)"')
_NAME_XML = re.compile(r"<name>([^<]+)</name>")
_NAME_ATTR = re.compile(r'<tool\s+name="([^"]+)"')
_NAME_DSML = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"')
_NAME_BARE = {"bash", "edit", "read", "write", "grep", "glob"}

_ARG_JSON = re.compile(r'"arguments"\s*:\s*(\{.*\})\s*$', re.DOTALL)
_ARG_PARAM = re.compile(
    r'<parameter\s+name="([^"]+)"[^>]*>\n?(.*?)</parameter>', re.DOTALL
)
_ARG_DSML = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"[^>]*>\n?(.*?)</｜DSML｜parameter>',
    re.DOTALL,
)
_ARG_XML_KV = re.compile(r"<([^<>/]+)>([^<]*?)</\1>", re.DOTALL)

# JSON tool object anywhere (for well-formed case 1)
_TOOL_JSON_RE = re.compile(
    r'<tool[^>]*>\s*(\{.*?"name".*?\})\s*(?:</arguments>\s*)?(?:</tool|</invoke|$)',
    re.DOTALL,
)


class ToolCall:
    """A parsed tool call (OpenAI-shaped at the boundary)."""

    def __init__(self, name: str, arguments: dict):
        self.id = "call_" + uuid.uuid4().hex[:24]
        self.name = name
        self.arguments = arguments

    def to_openai(self, index: int) -> dict:
        return {
            "index": index,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def _canonical(name: str, known: List[str]) -> Optional[str]:
    """Match a model-emitted tool name (any case) to a known tool name."""
    if name in known:
        return name
    low = name.lower().strip()
    for k in known:
        if k.lower() == low:
            return k
    return None


def _extract_json_args(region: str) -> Optional[dict]:
    m = _ARG_JSON.search(region)
    if not m:
        # whole region might be the JSON object
        try:
            obj = json.loads(region.strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return None
    raw = m.group(1)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # truncated JSON: salvage k:v pairs via regex
    salvaged = {}
    for km in re.finditer(r'"([^"]+)"\s*:\s*(?:"([^"]*)"|(\S+?)[,}\s])', raw):
        key = km.group(1)
        val = km.group(2) if km.group(2) is not None else km.group(3)
        if val is not None:
            salvaged[key] = val.strip('"')
    return salvaged or None


def _find_region_bounds(text: str) -> Tuple[int, int]:
    """(start, end_exclusive) of the tool-call region, or (-1, -1)."""
    first = None
    for o in _OPENERS:
        m = o.search(text)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), m.end())
    if first is None:
        return -1, -1
    last_end = first[1]
    for c in _CLOSERS:
        for m in c.finditer(text, first[1]):
            e = m.end()
            if e > last_end:
                last_end = e
    return first[0], last_end


def parse_tool_calls(
    text: str, known: List[str]
) -> Tuple[List[ToolCall], str]:
    """Extract structured tool calls from model output.

    `known` is the list of tool names the REQUEST advertised. Unknown names
    are still parsed if they look like tool calls (name extraction succeeds)
    — no: unknown names are treated as NOT a tool call (never guess), except
    case-insensitive matches.
    Returns (calls, clean_text) with all recognized markup removed.
    """
    if "<tool" not in text and "DSML" not in text and not any(
        f"<{t}>" in text for t in _NAME_BARE
    ):
        return [], text

    calls: List[ToolCall] = []
    clean = text

    # Pass 1: well-formed JSON blocks (case 1/3) — parse & remove
    for m in list(_TOOL_JSON_RE.finditer(clean)):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = _canonical(obj.get("name", ""), known) if obj.get("name") else None
        if not name:
            continue
        args = obj.get("arguments") or obj.get("parameters") or obj.get("input")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name, args))
    clean = _TOOL_JSON_RE.sub("", clean)

    # Pass 2: structural region parse for everything else
    while True:
        start, end = _find_region_bounds(clean)
        if start < 0:
            break
        region = clean[start:end]
        # inner = region minus the outer opener/closer tags, so inner k:v
        # extraction can't match the OUTER tags themselves
        inner = re.sub(r"^[^>]*>", "", region, count=1)
        inner = re.sub(r"<[^<>]*$", "", inner)  # drop any trailing partial tag
        inner = re.sub(r"</(tool|invoke|arguments)\s*>?\s*$", "", inner)
        # name
        name = None
        for pat in (_NAME_JSON, _NAME_XML, _NAME_ATTR, _NAME_DSML):
            m = pat.search(region)
            if m:
                name = _canonical(m.group(1), known)
                if name:
                    break
        if not name:
            # <parameter name="name">bash</parameter> style
            nm = re.search(r'<parameter\s+name="(?:name|tool)"[^>]*>\n?(\w[\w.-]*)\s*</parameter>', region)
            if nm:
                name = _canonical(nm.group(1), known)
        if not name:
            for bare in _NAME_BARE:
                if f"<{bare}>" in region and bare in known:
                    name = bare
                    break
        if not name and "command" in region and "bash" in known:
            # <tool><parameter name="command">...</parameter></tool> with no
            # name anywhere: the only tool with a plain 'command' arg
            name = "bash"
        if not name:
            # infer from a distinctive parameter name -> known tool
            # (order matters: more-specific signatures first; a 'content'
            # param with file_path means write, file_path alone means read)
            has_content = 'name="content"' in region
            has_path = 'name="file_path"' in region or 'name="path"' in region
            if "write" in known and has_content and has_path:
                name = "write"
            elif "edit" in known and any(f'name="{p}"' in region for p in ("old_string", "new_string")):
                name = "edit"
            if not name:
                for ptool, pnames in (("todowrite", ("todos",)),
                                      ("task", ("prompt", "description", "subagent")),
                                      ("webfetch", ("url", "format")),
                                      ("read", ("file_path", "path")),
                                      ("grep", ("pattern", "include")),
                                      ("glob", ("pattern", "glob")),
                                      ("skill", ("name",))):
                    if ptool in known and any(f'name="{p}"' in region for p in pnames):
                        name = ptool
                        break
        # arguments — priority: <parameter> tags, DSML params, bare-tag
        # body, simple XML k:v, then JSON salvage
        args: dict = {}
        pm = _ARG_PARAM.findall(region)
        if pm:
            args = {k.strip(): v.strip() for k, v in pm if k.strip() not in ("name", "tool", "description")}
        if not args:
            dm = _ARG_DSML.findall(region)
            if dm:
                args = {k.strip(): v.strip() for k, v in dm if k.strip() not in ("description",)}
        if not args:
            bm = re.search(r"<(bash|edit|read|write|grep|glob)>(.*?)</\1>", region, re.DOTALL)
            if bm:
                body = bm.group(2).strip()
                args = {"command": body} if bm.group(1) == "bash" else {"input": body}
        if not args:
            # simple XML k:v INSIDE the region (e.g. <command>x</command>)
            # skip outer-ish tags by only matching known-ish or short keys
            for k, v in _ARG_XML_KV.findall(inner):
                k, v = k.strip(), v.strip()
                if k in ("name", "tool", "arguments", "invoke", "parameter", "tool", "description"):
                    continue
                if "<" in v:  # nested — not a leaf value
                    continue
                if k:
                    args[k] = v
        if not args:
            jm = _extract_json_args(region)
            if isinstance(jm, dict):
                jm.pop("name", None)
                args = jm
        if not args or args == {}:
            # bare JSON object inside the region is the arguments
            # (e.g. <tool name="bash">{"command": "ls"}</tool>)
            for jm2 in re.finditer(r"\{([^{}]*)\}", region):
                try:
                    obj = json.loads("{" + jm2.group(1) + "}")
                    if isinstance(obj, dict) and obj:
                        obj.pop("name", None)
                        if obj:
                            args = obj
                            break
                except json.JSONDecodeError:
                    continue
        if name:
            calls.append(ToolCall(name, args or {}))
        clean = clean[:start] + clean[end:]
        # continue scanning (multiple regions)

    # Cleanup pass: strip leftover fragments so nothing leaks as content
    clean = re.sub(r"</?(tool|invoke|arguments|parameter|name)\b[^>]*>?\s*", "", clean)
    clean = re.sub(r"</?｜DSML｜[^>]*>?\s*", "", clean)
    clean = re.sub(r"<tool\s*$", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    return calls, clean


def contains_tool_markup(text: str) -> bool:
    """Any tool-call opener — used to trigger stream holdback."""
    if "<tool" in text or "｜DSML｜" in text:
        return True
    return any(f"<{t}>" in text for t in _NAME_BARE)


def build_tool_instructions(tools: List[dict]) -> str:
    """Render the [AVAILABLE TOOLS] + [TOOL CALLING INSTRUCTION] block."""
    if not tools:
        return ""
    compact = []
    for t in tools:
        fn = t.get("function", t)
        compact.append({
            "name": fn.get("name"),
            "description": (fn.get("description") or "")[:400],
            "parameters": fn.get("parameters", {}),
        })
    return (
        "\n\n[AVAILABLE TOOLS]\n"
        + json.dumps(compact, ensure_ascii=False, indent=1)
        + "\n\n[TOOL CALLING INSTRUCTION]\n"
        "To call a tool, output a JSON block wrapped in <tool>...</tool> tags:\n"
        '<tool>{"name": "tool_name", "arguments": {"arg": "value"}}</tool>\n'
        "Rules:\n"
        "- Output ONLY the <tool>...</tool> block(s) when calling a tool.\n"
        "- If no tool is needed, respond with standard answer text.\n"
        "- Never invent tools that are not in the list above.\n"
    )


def format_tool_result(name: str, content: str) -> str:
    """Render a tool result message for the DeepSeek prompt."""
    return f"Tool result ({name}):\n{content}"
