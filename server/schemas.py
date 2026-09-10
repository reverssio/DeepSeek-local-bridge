"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style). We only
    # read text parts; non-text parts are ignored.
    content: Union[str, List[dict], None] = None
    # assistant tool-call echo (OpenAI standard shape)
    tool_calls: Optional[List[dict]] = None
    # tool role message fields
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # Reasoning echo: with interleaved:{field:"reasoning_content"} configured,
    # OpenCode replays past assistant reasoning back on the assistant message.
    # We accept and (see openai_format) forward it so DeepSeek sees its own
    # prior reasoning during tool-call trajectories.
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # OpenAI tools (function definitions) the caller wants the model to use.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    # DeepSeek toggles (extra_body extras)
    thinking: bool = False
    search: bool = False
    # OpenCode variant passthroughs (openai-compatible maps reasoningEffort
    # -> reasoning_effort and thinkingBudget -> thinking_budget in the body).
    # Any non-null value means "thinking requested".
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # NOTE: intentionally NOT enforced locally — the underlying service's
    # own limits are the only limits (no artificial truncation).
    max_tokens: Optional[int] = None
    user: Optional[str] = None
