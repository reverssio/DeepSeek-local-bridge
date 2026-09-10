"""Translate between OpenAI's chat-completions shapes and the DeepSeek bridge.

DeepSeek's protocol has a single `prompt` string per turn. The bridge:

  - builds prompts (first turn: full history + tool instructions; continuation
    turns: only the NEW messages — the DeepSeek conversation already holds the
    earlier ones natively),
  - wraps the typed upstream events (content / reasoning) back into OpenAI
    response objects, keeping reasoning in `reasoning_content` (the field
    DeepSeek's own official API uses for DeepThink output) and tool calls in
    `tool_calls` / streaming tool-call deltas.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List

from .schemas import ChatMessage
from .tools_bridge import (
    ToolCall,
    build_tool_instructions,
    format_tool_result,
    get_tool_nonce,
)

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt (used for NEW conversations).

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    tool_id_to_name = {}
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                tid = tc.get("id")
                fn = tc.get("function", {}).get("name")
                if tid and fn:
                    tool_id_to_name[tid] = fn

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        text = _text_of(m.content)
        # assistant tool-call echo -> render back into the instructed markup so
        # the model sees its own request shape
        if m.role == "assistant" and m.tool_calls:
            # prior reasoning (replayed by OpenCode with interleaved
            # reasoning_content) is preserved so the model sees its own
            # thought process across tool rounds
            if getattr(m, "reasoning_content", None):
                lines.append(f"Assistant (thinking): {m.reasoning_content}")
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"input": fn.get("arguments")}
                lines.append(f"Assistant: <tool>{{\"name\": {json.dumps(fn.get('name'))}, "
                             f"\"arguments\": {json.dumps(args, ensure_ascii=False)}}}</tool>")
            if text:
                lines.append(f"Assistant: {text}")
            continue
        if m.role == "tool":
            tname = m.name or tool_id_to_name.get(m.tool_call_id) or "tool"
            lines.append(f"User: {format_tool_result(tname, text)}")
            continue
        lines.append(f"{label}: {text}")
    if messages and messages[-1].role != "assistant":
        lines.append("Assistant:")
    return "\n\n".join(lines)


def new_turn_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """The trailing messages that are NEW since the mapped conversation turn.

    Continuation requests contain the full history; the DeepSeek conversation
    already holds everything up to and including the assistant reply we
    returned. The new information is: tool-result messages and/or the new
    user message (everything after the last assistant echo). We detect it by
    walking back to the last assistant message.
    """
    last_assistant = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_assistant = i
            break
    if last_assistant < 0:
        return messages
    return messages[last_assistant + 1:]


def build_first_prompt(messages: List[ChatMessage], tools: list) -> str:
    """Prompt for a NEW DeepSeek conversation: history + tool instructions.

    The tool instructions are placed BEFORE the trailing 'Assistant:' cue so
    the model reads them as rules for ITS next turn, not as content to
    answer about.
    """
    base = messages_to_prompt(messages)
    nonce = get_tool_nonce(tools) if tools else None
    instr = build_tool_instructions(tools or [], nonce=nonce)
    if not instr:
        return base
    # Move the trailing 'Assistant:' cue to after the instructions.
    cue = "\n\nAssistant:"
    if base.endswith(cue):
        return base[: -len(cue)].rstrip() + instr + cue
    return base + instr


def build_continuation_prompt(new_msgs: List[ChatMessage], tools: list,
                              used_tool_call: bool,
                              all_messages: Optional[List[ChatMessage]] = None) -> str:
    """Prompt for a CONTINUING DeepSeek conversation: only the new information.

    Tool results (role=tool) are rendered with their canonical tool name;
    a fresh user message is labelled User; tool definitions and instructions
    are preserved so the model retains tool schemas and invocation contracts.
    """
    tool_id_to_name = {}
    if all_messages:
        for m in all_messages:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    tid = tc.get("id")
                    fn = tc.get("function", {}).get("name")
                    if tid and fn:
                        tool_id_to_name[tid] = fn

    parts = []
    has_tool_result = False
    for m in new_msgs:
        text = _text_of(m.content)
        if m.role == "tool":
            has_tool_result = True
            tname = m.name or tool_id_to_name.get(m.tool_call_id) or "tool"
            parts.append(format_tool_result(tname, text))
        elif m.role == "user":
            parts.append(f"User: {text}")
        elif m.role == "assistant":
            # (assistant text between tool rounds — rare; include verbatim)
            if text or m.tool_calls:
                rendered = messages_to_prompt([m])
                parts.append(rendered)
        # system messages on continuation: ignore (the conversation already
        # has the system context from turn 1; OpenCode repeats it verbatim)

    if has_tool_result:
        parts.append(
            "Continue the task using the tool results above. Do NOT repeat tool calls that already "
            "succeeded; perform the next step or give the final answer."
        )

    out = "\n\n".join(p for p in parts if p)
    if tools:
        nonce = get_tool_nonce(tools) if tools else None
        instr = build_tool_instructions(tools, nonce=nonce)
        if instr:
            out += instr
    return out


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None,
                        reasoning: str = "",
                        tool_calls: List[ToolCall] = None) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    `conversation_id` is an extra top-level field (outside OpenAI's schema) you
    can send back to resume the conversation. `reasoning_content` carries the
    DeepThink reasoning (same field DeepSeek's official API uses). `tool_calls`
    carries parsed structured tool calls.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(content or "")
    message: dict = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [tc.to_openai(i) for i, tc in enumerate(tool_calls)]
        message.pop("content", None) if content in (None, "") else None
        if not content:
            message["content"] = None
    finish = "tool_calls" if tool_calls else "stop"
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(model: str, stream) -> Iterable[str]:
    """Yield OpenAI SSE lines for a streamed completion.

    `stream` yields typed events: ("content", delta) | ("reasoning", delta) |
    ("tool_calls", [ToolCall]) | ("error", msg). Reasoning deltas use
    `reasoning_content` (DeepSeek-API standard); tool calls are emitted as
    OpenAI streaming tool_call deltas with `finish_reason: "tool_calls"`.
    The stream's `.conversation_id` is attached to the final chunk.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})

    saw_error = False
    tc_index = 0
    for kind, value in stream:
        if kind == "content" and value:
            yield frame({"content": value})
        elif kind == "reasoning" and value:
            yield frame({"reasoning_content": value})
        elif kind == "tool_calls" and value:
            # value: list[ToolCall]; emit as streaming deltas with monotonic sequential index
            for tc in value:
                yield frame({"tool_calls": [tc.to_openai(tc_index)]})
                tc_index += 1
        elif kind == "error" and value:
            # upstream/protocol failure mid-stream: end the stream with an
            # OpenAI-style error frame; nothing else follows
            saw_error = True
            err = {"error": {"message": str(value), "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

    conversation_id = getattr(stream, "conversation_id", None)
    finish = getattr(stream, "finish_reason", "stop")
    yield frame({}, finish=finish, extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
