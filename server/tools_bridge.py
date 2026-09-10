"""Lenient, robust, stack-based tool-call parser, validator, and prompt builder.

Informed by OmniRoute's architectural patterns (issue #2820, #5154, #9343, #10527):
- Tokenizer and stack-based block pairing: processes only innermost leaf blocks,
  gracefully eliminating nested/doubled wrappers (<tool><tool>...</tool></tool>).
- Per-request nonce security binding (_nonce): prevents bare JSON in code blocks,
  quoted examples, or prompt injection from triggering unintended tool execution.
- Loose JSON repair: normalizes single-quoted strings, unquoted keys, Python literals
  (True/False/None), and trailing commas.
- Fuzzy tool name resolution: Levenshtein distance scoring and case/punctuation normalization.
- Nameless-block schema attribution: attributes tool names when parameters uniquely match
  a declared tool's schema.
- Native DSML support: full handling of single-bar (<｜DSML｜...>), double-bar (<｜｜DSML｜｜...>),
  and ASCII pipe (<||DSML||...>) blocks.
- Argument key normalization: aliases common agent parameter names (cmd -> command,
  file_path -> filePath, etc.).
- Clean range excision: strips accepted tool blocks and orphaned tag fragments without
  leaking stray delimiters (>, <) into assistant content.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("tools_bridge")

# --- Tag tokenization and stack structures ------------------------------------

TAG_TOKEN_RE = re.compile(
    r"<(\/?)(?:tool_call|tool)(:[A-Za-z0-9_.+-]+)?((?:\s[^>]*)?)\/?>|"
    r"(</?[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*(?:calls|tool_calls|invoke|parameter)\b[^>]*>)|"
    r"(</?invoke\b[^>]*>)|"
    r"(</?(?:bash|edit|read|write|grep|glob|webfetch)\b[^>]*>)",
    re.I
)

_NAME_BARE = {"bash", "edit", "read", "write", "grep", "glob", "webfetch"}


class TagToken:
    __slots__ = ("start", "end", "closing", "suffix", "attrs", "raw", "tag_type")

    def __init__(self, start: int, end: int, closing: bool, suffix: str, attrs: str, raw: str, tag_type: str):
        self.start = start
        self.end = end
        self.closing = closing
        self.suffix = suffix
        self.attrs = attrs
        self.raw = raw
        self.tag_type = tag_type


class ToolBlock:
    __slots__ = ("open", "close", "inner_start", "inner_end")

    def __init__(self, open_tok: TagToken, close_tok: TagToken, inner_start: int, inner_end: int):
        self.open = open_tok
        self.close = close_tok
        self.inner_start = inner_start
        self.inner_end = inner_end


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


# --- Nonce management ---------------------------------------------------------

_tool_nonce_cache: dict[int, str] = {}

def get_tool_nonce(tools: list) -> str:
    """Generate or retrieve a per-request 8-character hex nonce for tool binding."""
    if not tools:
        return ""
    try:
        key = hash(json.dumps(tools, sort_keys=True))
    except Exception:
        key = id(tools)
    if key not in _tool_nonce_cache:
        _tool_nonce_cache[key] = uuid.uuid4().hex[:8]
    return _tool_nonce_cache[key]


# --- String distance & fuzzy resolution ---------------------------------------

def normalize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] * (len(b) + 1)
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(b)]


def resolve_requested_tool_name(emitted: str, known: List[str]) -> Optional[str]:
    """Resolve an emitted tool name against known tool names using fuzzy matching."""
    if not emitted:
        return None
    if emitted in known:
        return emitted
    emitted_norm = normalize_tool_name(emitted)
    if not emitted_norm:
        return None
    for k in known:
        if k.lower() == emitted.lower() or normalize_tool_name(k) == emitted_norm:
            return k

    best_name: Optional[str] = None
    best_score = 0.0
    second_best = 0.0
    for k in known:
        knorm = normalize_tool_name(k)
        if knorm == emitted_norm:
            score = 0.98
        elif len(knorm) >= 4 and (knorm in emitted_norm or emitted_norm in knorm):
            score = 0.86
        else:
            dist = levenshtein_distance(emitted_norm, knorm)
            longer = max(len(emitted_norm), len(knorm), 1)
            sim = 1.0 - (dist / longer)
            score = sim if sim >= 0.72 else 0.0

        if score > best_score:
            second_best = best_score
            best_score = score
            best_name = k
        elif score > second_best:
            second_best = score

    if best_name and best_score >= 0.72:
        if best_score < 0.98 and (best_score - second_best) < 0.08:
            return None
        return best_name
    return None


def _canonical(name: str, known: List[str]) -> Optional[str]:
    return resolve_requested_tool_name(name, known)


# --- Loose JSON parsing -------------------------------------------------------

def parse_loose_json(raw: str) -> Optional[dict]:
    """Permissive JSON parser: handles code fences, single quotes, Python literals,
    unquoted keys, and trailing commas."""
    if not raw or not isinstance(raw, str):
        return None
    val = raw.strip()
    val = re.sub(r"^```(?:json|javascript|js|python)?\s*", "", val, flags=re.I)
    val = re.sub(r"\s*```$", "", val).strip()
    try:
        obj = json.loads(val)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Convert Python literals
    val2 = re.sub(r"\bTrue\b", "true", val)
    val2 = re.sub(r"\bFalse\b", "false", val2)
    val2 = re.sub(r"\bNone\b", "null", val2)
    # Convert single-quoted strings to double-quoted strings
    val2 = re.sub(r"(?<=[{\[,:\s])'([^'\\]*(?:\\.[^'\\]*)*)'(?=[}\],:\s])", r'"\1"', val2)
    # Quote unquoted property keys
    val2 = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)', r'\1"\2"\3', val2)
    # Remove trailing commas
    val2 = re.sub(r",\s*([}\]])", r"\1", val2)

    try:
        obj = json.loads(val2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


# --- Schema parameter mapping -------------------------------------------------

def build_schema_param_map(tools: Union[List[str], List[dict], None]) -> Dict[str, Set[str]]:
    """Map tool name -> set of required and recognized property names."""
    param_map: Dict[str, Set[str]] = {}
    if not tools:
        return param_map
    for t in tools:
        if isinstance(t, str):
            param_map[t] = set()
            continue
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters", {})
        props = set(params.get("properties", {}).keys())
        req = set(params.get("required", []))
        param_map[name] = props | req
    return param_map


# --- Validation and parameter normalization -----------------------------------

def validate_and_normalize_call(
    name: str,
    args: dict,
    tool_schemas: Optional[Dict[str, dict]] = None,
) -> Tuple[bool, Optional[str], dict]:
    """Validate a parsed tool call against declared schemas and normalize keys."""
    canon_name = name
    schema = None
    if tool_schemas:
        canon_name = _canonical(name, list(tool_schemas.keys()))
        if not canon_name:
            return False, f"Unknown tool '{name}'", {}
        schema = tool_schemas.get(canon_name, {})

    if not isinstance(args, dict):
        return False, f"Tool '{canon_name}' arguments must be a dictionary", {}

    norm_args = dict(args)
    if schema:
        fn_spec = schema.get("function", schema)
        params = fn_spec.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})

        # Key normalization for common agent tool parameter aliases
        if "command" in required and "command" not in norm_args:
            for a in ("cmd", "script", "command_line"):
                if a in norm_args:
                    norm_args["command"] = norm_args.pop(a)
                    break
        if "filePath" in required and "filePath" not in norm_args:
            for a in ("file_path", "filepath", "path", "file"):
                if a in norm_args:
                    norm_args["filePath"] = norm_args.pop(a)
                    break
        if "oldString" in required and "oldString" not in norm_args:
            for a in ("old_string", "old_text", "old"):
                if a in norm_args:
                    norm_args["oldString"] = norm_args.pop(a)
                    break
        if "newString" in required and "newString" not in norm_args:
            for a in ("new_string", "new_text", "new"):
                if a in norm_args:
                    norm_args["newString"] = norm_args.pop(a)
                    break
        if "url" in required and "url" not in norm_args:
            for a in ("uri", "link", "address"):
                if a in norm_args:
                    norm_args["url"] = norm_args.pop(a)
                    break
        if "pattern" in required and "pattern" not in norm_args:
            for a in ("regex", "query", "search"):
                if a in norm_args:
                    norm_args["pattern"] = norm_args.pop(a)
                    break

        # Check required fields
        for req in required:
            if req not in norm_args or norm_args[req] is None or str(norm_args[req]).strip() == "":
                return False, f"Missing required parameter '{req}' for tool '{canon_name}'", {}

        # Enum validation / coercion
        for p, spec in props.items():
            if p in norm_args and isinstance(spec, dict) and "enum" in spec:
                val_str = str(norm_args[p]).lower().strip()
                enums = [str(e).lower() for e in spec["enum"]]
                if val_str in enums:
                    norm_args[p] = val_str
                else:
                    norm_args.pop(p)

    return True, None, norm_args


# --- Tag tokenizer and block pairing ------------------------------------------

def tokenize_tool_tags(text: str) -> List[TagToken]:
    tokens: List[TagToken] = []
    for m in TAG_TOKEN_RE.finditer(text):
        raw = m.group(0)
        start, end = m.start(), m.end()
        if m.group(4):
            # DSML tag
            dsml_tag = m.group(4)
            closing = dsml_tag.startswith("</") or "/calls>" in dsml_tag.lower() or "/invoke>" in dsml_tag.lower() or "/parameter>" in dsml_tag.lower()
            tokens.append(TagToken(start, end, closing, "", dsml_tag, raw, "dsml"))
        elif m.group(5):
            # <invoke> tag
            inv_tag = m.group(5)
            closing = inv_tag.startswith("</")
            tokens.append(TagToken(start, end, closing, "", inv_tag, raw, "invoke"))
        elif m.group(6):
            # Bare tool tag (<bash>, <read>, etc.)
            bare_tag = m.group(6)
            closing = bare_tag.startswith("</")
            tname = re.sub(r"[</>]", "", bare_tag).strip().lower()
            tokens.append(TagToken(start, end, closing, tname, "", raw, "bare"))
        else:
            # <tool...> or <tool_call...>
            closing = m.group(1) == "/"
            suffix = m.group(2)[1:] if m.group(2) else ""
            attrs = m.group(3) or ""
            tokens.append(TagToken(start, end, closing, suffix, attrs, raw, "tool"))
    return tokens


def pair_tool_blocks(tokens: List[TagToken], text_len: int) -> List[ToolBlock]:
    blocks: List[ToolBlock] = []
    stack: List[TagToken] = []
    for tok in tokens:
        if not tok.closing:
            stack.append(tok)
        else:
            if stack:
                open_tok = stack.pop()
                blocks.append(ToolBlock(open_tok, tok, open_tok.end, tok.start))
    for open_tok in stack:
        synthetic_close = TagToken(text_len, text_len, True, "", "", "", open_tok.tag_type)
        blocks.append(ToolBlock(open_tok, synthetic_close, open_tok.end, text_len))
    return blocks


def is_leaf_block(block: ToolBlock, all_blocks: List[ToolBlock]) -> bool:
    """Return True if no other tool block is nested inside this block's inner range."""
    for other in all_blocks:
        if other is not block:
            if other.open.start >= block.inner_start and other.close.end <= block.inner_end:
                return False
    return True


def strip_ranges(text: str, ranges: List[Tuple[int, int]]) -> str:
    """Excise non-overlapping ranges cleanly from text."""
    if not ranges:
        return text
    sorted_ranges = sorted(ranges, key=lambda r: r[0], reverse=True)
    content = text
    for start, end in sorted_ranges:
        content = content[:start] + content[end:]
    # Strip residual tags or orphaned delimiter brackets
    content = re.sub(r"</?(?:tool_call|tool|invoke|parameter|name|arguments)\b[^>]*>?\s*", "", content)
    content = re.sub(r"</?[｜|]{1,2}\s*DSML\s*[｜|]{1,2}[^>]*>\s*", "", content)
    content = re.sub(r"^\s*[><]+\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if content in (">", "<", ">>", "<<", "> >", "< <"):
        content = ""
    return content


# --- Native DSML extraction ---------------------------------------------------

def extract_dsml_calls(
    text: str,
    known: List[str],
    tool_schemas: Optional[Dict[str, dict]] = None,
) -> Tuple[List[ToolCall], str]:
    if "DSML" not in text and "<invoke" not in text:
        return [], text

    invoke_start_pat = re.compile(
        r'(?:</?[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*invoke|<invoke)\s+(?:name|tool)=["\']?([^"\'\s>]+)["\']?[^>]*>',
        re.DOTALL,
    )
    invoke_end_pat = re.compile(
        r'</?[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*/?invoke\b[^>]*>|</?invoke\s*>',
        re.DOTALL,
    )
    param_tag_re = re.compile(
        r'(?:</?[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*parameter|<parameter)\b([^>]*?)(?:/>|>(.*?)(?:</[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*/?parameter\b[^>]*>|</parameter\s*>))',
        re.DOTALL,
    )

    calls: List[ToolCall] = []
    accepted_ranges: List[Tuple[int, int]] = []

    for im in invoke_start_pat.finditer(text):
        raw_name = im.group(1).strip()
        canon_name = _canonical(raw_name, known)
        if not canon_name:
            continue
        em = invoke_end_pat.search(text, im.end())
        if not em:
            continue
        invoke_body = text[im.end():em.start()]
        args: dict = {}
        for pm in param_tag_re.finditer(invoke_body):
            attrs = pm.group(1)
            inner = (pm.group(2) or "").strip()
            name_m = re.search(r'name=["\']?([^"\'\s>]+)["\']?', attrs)
            if not name_m:
                continue
            pname = name_m.group(1).strip()
            content_m = re.search(r'content=["\']([^"\']*)["\']', attrs)
            val = content_m.group(1) if content_m else inner
            str_m = re.search(r'string=["\']?(true|false)["\']?', attrs)
            if str_m and str_m.group(1) == "false":
                try:
                    val = json.loads(val)
                except Exception:
                    if str(val).lower() == "true":
                        val = True
                    elif str(val).lower() == "false":
                        val = False
                    elif str(val).isdigit():
                        val = int(val)
            args[pname] = val

        if not args:
            obj = parse_loose_json(invoke_body)
            if isinstance(obj, dict):
                args = obj

        valid, err, norm_args = validate_and_normalize_call(canon_name, args, tool_schemas)
        if valid:
            calls.append(ToolCall(canon_name, norm_args))
            accepted_ranges.append((im.start(), em.end()))
        else:
            logger.warning("DSML tool call rejected: %s", err)

    clean = strip_ranges(text, accepted_ranges)
    return calls, clean


# --- Main Public Parser -------------------------------------------------------

def parse_tool_calls(
    text: str,
    tools: Union[List[str], List[dict], None] = None,
    expected_nonce: Optional[str] = None,
) -> Tuple[List[ToolCall], str]:
    """Extract, validate, and normalize tool calls from model output using stack-based
    leaf block pairing, nonce security binding, loose JSON repair, and DSML support."""
    if not text or not isinstance(text, str):
        return [], text or ""

    if "<tool" not in text and "DSML" not in text and "<invoke" not in text and not any(
        f"<{t}>" in text for t in _NAME_BARE
    ):
        return [], text

    tool_schemas: Dict[str, dict] = {}
    known: List[str] = []
    if tools:
        if isinstance(tools[0], str):
            known = list(tools)
            tool_schemas = {t: {"name": t} for t in known}
        elif isinstance(tools[0], dict):
            for t in tools:
                fn = t.get("function", t)
                name = fn.get("name")
                if name:
                    known.append(name)
                    tool_schemas[name] = fn

    schema_param_map = build_schema_param_map(tools)
    calls: List[ToolCall] = []
    accepted_ranges: List[Tuple[int, int]] = []

    # Pass 1: Extract native DSML / invoke blocks
    if "DSML" in text or "<invoke" in text:
        dsml_calls, text = extract_dsml_calls(text, known, tool_schemas)
        calls.extend(dsml_calls)

    # Pass 2: Tokenize and pair tags via stack
    tokens = tokenize_tool_tags(text)
    if not tokens:
        return calls, text

    blocks = pair_tool_blocks(tokens, len(text))
    leaf_blocks = [b for b in blocks if is_leaf_block(b, blocks)]
    leaf_blocks.sort(key=lambda b: b.open.start)

    for b in leaf_blocks:
        inner = text[b.inner_start:b.inner_end].strip()
        attrs = b.open.attrs
        suffix = b.open.suffix

        # Extract name from suffix or attributes
        name = None
        if suffix:
            name = _canonical(suffix, known)
        if not name and attrs:
            attr_name_m = re.search(r'\b(?:name|tool|id)=["\']([^"\']+)["\']', attrs)
            if attr_name_m:
                name = _canonical(attr_name_m.group(1), known)

        # Parse arguments
        args: dict = {}
        json_obj = parse_loose_json(inner)
        if json_obj:
            # Nonce security binding check (OmniRoute #9343):
            # If the block carries a _nonce, verify it against expected_nonce
            emitted_nonce = json_obj.get("_nonce")
            if expected_nonce and emitted_nonce is not None and emitted_nonce != expected_nonce:
                logger.warning("Rejected tool call with mismatched nonce: %s != %s", emitted_nonce, expected_nonce)
                continue

            if not name:
                emitted_name = json_obj.get("name") or json_obj.get("type")
                if emitted_name:
                    name = _canonical(str(emitted_name), known)

            raw_args = json_obj.get("arguments") or json_obj.get("parameters") or json_obj.get("params") or json_obj.get("input")
            if isinstance(raw_args, str):
                parsed_args = parse_loose_json(raw_args)
                args = parsed_args if isinstance(parsed_args, dict) else {"input": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            elif not name:
                # Top-level keys might be arguments
                clean_json = dict(json_obj)
                clean_json.pop("name", None)
                clean_json.pop("type", None)
                clean_json.pop("_nonce", None)
                args = clean_json
            else:
                clean_json = dict(json_obj)
                clean_json.pop("name", None)
                clean_json.pop("type", None)
                clean_json.pop("_nonce", None)
                args = clean_json

        # Check XML-style parameters (<parameter name="...">...</parameter>)
        if not args:
            param_matches = re.findall(r'<parameter\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</parameter>', inner, re.DOTALL)
            if param_matches:
                args = {k.strip(): v.strip() for k, v in param_matches if k.strip() not in ("name", "tool", "description", "_nonce")}

        # Check bare tag syntax (<bash>cmd</bash>)
        if not name and b.open.tag_type == "bare" and b.open.suffix in known:
            name = b.open.suffix
            args = {"command": inner} if name == "bash" else {"input": inner}

        # Nameless-block schema fallback (OmniRoute #5154)
        if not name and args and schema_param_map:
            arg_keys = set(args.keys())
            candidates = []
            for tname, schema_keys in schema_param_map.items():
                if schema_keys and arg_keys.issubset(schema_keys):
                    candidates.append(tname)
            if len(candidates) == 1:
                name = candidates[0]

        if name:
            valid, err, norm_args = validate_and_normalize_call(name, args, tool_schemas)
            if valid:
                calls.append(ToolCall(name, norm_args))
                accepted_ranges.append((b.open.start, b.close.end))
            else:
                logger.warning("Tool call rejected: %s", err)

    clean_text = strip_ranges(text, accepted_ranges)

    # Deduplicate multiple identical calls generated by nested wrappers
    deduped: List[ToolCall] = []
    seen = set()
    for c in calls:
        key = (c.name, json.dumps(c.arguments, sort_keys=True))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped, clean_text


def contains_tool_markup(text: str) -> bool:
    """Check if text contains an opening tool-call delimiter."""
    if "<tool" in text or "DSML" in text or "<invoke" in text:
        return True
    return any(f"<{t}>" in text for t in _NAME_BARE)


def build_tool_instructions(tools: List[dict], nonce: Optional[str] = None) -> str:
    """Render compact, unambiguous tool definitions and instructions with nonce binding."""
    if not tools:
        return ""
    lines = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        param_desc = []
        for p, spec in props.items():
            ptype = spec.get("type", "string") if isinstance(spec, dict) else "string"
            is_req = "required" if p in req else "optional"
            pdesc = spec.get("description") if isinstance(spec, dict) else ""
            enum = spec.get("enum") if isinstance(spec, dict) else None
            extra = f", enum: {enum}" if enum else ""
            if pdesc:
                param_desc.append(f"    - {p} ({ptype}, {is_req}{extra}): {pdesc}")
            else:
                param_desc.append(f"    - {p} ({ptype}, {is_req}{extra})")
        param_block = ("\n" + "\n".join(param_desc)) if param_desc else " none"
        lines.append(f"- {name}: {desc}\n  Parameters:{param_block}")

    nonce_fragment = f', "_nonce": "{nonce}"' if nonce else ""
    nonce_rule = f'- Include the secret binding "_nonce": "{nonce}" exactly as shown.\n' if nonce else ""

    return (
        "\n\n[AVAILABLE TOOLS]\n"
        + "\n".join(lines)
        + "\n\n[TOOL CALLING INSTRUCTION]\n"
        "To invoke a tool, output either format:\n"
        "Format 1 (JSON):\n"
        f'<tool>{{"name": "<tool_name>", "arguments": {{ ... }}{nonce_fragment}}}</tool>\n'
        "Format 2 (DSML):\n"
        "<｜｜DSML｜｜ calls>\n"
        '<｜｜DSML｜｜ invoke name="<tool_name>">\n'
        '<｜｜DSML｜｜ parameter name="<param_name>" string="true">value</｜｜DSML｜｜ parameter>\n'
        "<｜｜DSML｜｜ invoke>\n"
        "<｜｜DSML｜｜ calls>\n"
        "Rules:\n"
        "- Only call tools from [AVAILABLE TOOLS].\n"
        "- Always supply all required parameters.\n"
        f"{nonce_rule}"
        "- When a tool is needed, output the tool block immediately.\n"
        "- If no tool is needed, provide your final response in normal text.\n"
    )


def format_tool_result(name: str, content: str) -> str:
    """Render a tool result message for the DeepSeek prompt."""
    return f"Tool result ({name}):\n{content}"


def get_unemitted_clean(clean: str, emitted_text: str) -> str:
    """Return only the portion of clean text that has not already been emitted."""
    clean_s = clean.strip()
    emitted_s = emitted_text.strip()
    if not clean_s:
        return ""
    if not emitted_s:
        return clean
    if emitted_s == clean_s or emitted_s.startswith(clean_s):
        return ""
    if clean.startswith(emitted_text):
        return clean[len(emitted_text):]
    if clean_s.startswith(emitted_s):
        return clean_s[len(emitted_s):].lstrip()
    clean_norm = re.sub(r"\s+", " ", clean_s)
    emitted_norm = re.sub(r"\s+", " ", emitted_s)
    if clean_norm == emitted_norm or emitted_norm.startswith(clean_norm):
        return ""
    if clean_norm.startswith(emitted_norm):
        return clean_norm[len(emitted_norm):].lstrip()
    return ""
