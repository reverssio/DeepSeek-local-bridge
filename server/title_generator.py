"""Intelligent local title generator for OpenCode sessions.

OpenCode fires a background request with agent="title" after the first user turn.
The system prompt asks for:
- A single line, <= 50 characters, no explanations.
- Same language as user message.
- Grammatically correct, natural phrasing.
- No tool names (no "read tool", "bash tool", "edit tool").
- Focus on main topic or question.
- No repetitive prefixes like "Analyzing".
- For files, focus on WHAT the user wants to do WITH the file.
- Keep exact: technical terms, numbers, filenames, HTTP codes.
- Remove: the, this, my, a, an.
- If conversational (hello, hey): reflective title (Greeting, Quick check-in, Light chat).
"""

from __future__ import annotations

import re
from typing import List


# Stop words to remove from title starts
_LEADING_ARTICLES_RE = re.compile(r"^(?:the|this|my|a|an)\s+", re.IGNORECASE)

# Conversational phrases -> standard titles
_CONVERSATIONAL_MAP = {
    "hello": "Greeting",
    "hi": "Greeting",
    "hey": "Greeting",
    "good morning": "Morning greeting",
    "good afternoon": "Afternoon greeting",
    "good evening": "Evening greeting",
    "what's up": "Quick check-in",
    "whats up": "Quick check-in",
    "sup": "Quick check-in",
    "how are you": "Check-in",
    "who are you": "Identity inquiry",
    "what are you": "Capabilities inquiry",
    "help": "Help request",
    "test": "System test",
    "ping": "Ping test",
}


def _clean_text(text: str) -> str:
    """Strip markdown formatting, URLs, and excessive whitespace."""
    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove raw URLs
    text = re.sub(r"https?://[^\s]+", "", text)
    # Remove tool-related tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_user_query(messages: List[dict]) -> str:
    """Extract the primary user query text from the title request messages."""
    user_texts = []
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        if not isinstance(c, str):
            continue
        c = c.strip()
        # OpenCode's title prompt starts with: "Generate a title for this conversation:\n"
        prefix = "Generate a title for this conversation:"
        if c.startswith(prefix):
            c = c[len(prefix):].strip()
        if c:
            user_texts.append(c)

    if not user_texts:
        return ""
    # The last user text is usually the query to summarize
    return user_texts[-1]


def generate_title(messages: List[dict]) -> str:
    """Generate a clean, high-quality <= 50-character title from conversation messages."""
    raw_query = extract_user_query(messages)
    if not raw_query:
        return "New session"

    # Check for short conversational greetings
    low_clean = raw_query.lower().strip("?!., \t\n")
    if low_clean in _CONVERSATIONAL_MAP:
        return _CONVERSATIONAL_MAP[low_clean]

    for greeting, title in _CONVERSATIONAL_MAP.items():
        if low_clean.startswith(greeting) and len(low_clean) <= len(greeting) + 5:
            return title

    # Check for image inspection / visual analysis
    image_match = re.search(r"(/storage/[^\s\"'<>]+\.(?:jpg|jpeg|png|webp|gif|bmp)|images?\.(?:jpe?g|png|webp))", raw_query, re.I)
    if image_match or "image" in low_clean or "picture" in low_clean or "screenshot" in low_clean or "photo" in low_clean:
        if "analyze" in low_clean or "inspect" in low_clean or "examine" in low_clean:
            # Check if specific filename exists
            fn = image_match.group(1).split("/")[-1] if image_match else ""
            if fn and len(fn) <= 25:
                return f"Analyze image {fn}"[:50]
            return "Visual image analysis"
        if image_match:
            fn = image_match.group(1).split("/")[-1]
            return f"Image inspection: {fn}"[:50]
        return "Image analysis"

    # Check for code creation / script running
    script_match = re.search(r"([a-zA-Z0-9_\-]+\.(?:py|sh|js|ts|c|cpp|rs|java|kt|go))\b", raw_query, re.I)
    if script_match:
        script_name = script_match.group(1)
        if any(w in low_clean for w in ("create", "write", "make")) and any(w in low_clean for w in ("run", "execute")):
            return f"Create and run {script_name}"[:50]
        if any(w in low_clean for w in ("create", "write", "make")):
            return f"Create {script_name}"[:50]
        if any(w in low_clean for w in ("run", "execute", "test")):
            return f"Run {script_name}"[:50]
        if any(w in low_clean for w in ("debug", "fix", "repair")):
            return f"Debug {script_name}"[:50]

    # Check for repo / GitHub queries
    repo_match = re.search(r"github\.com/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)", raw_query, re.I)
    if repo_match:
        repo_name = repo_match.group(2)
        if "what" in low_clean or "understand" in low_clean or "about" in low_clean or "explain" in low_clean:
            return f"{repo_name} repository overview"[:50]
        return f"{repo_name} repository exploration"[:50]

    # Check for knowledge / capability inquiry
    if "scope of your knowledge" in low_clean or "knowledge cutoff" in low_clean or "what do you know" in low_clean:
        return "Knowledge scope inquiry"
    if "what models" in low_clean or "available models" in low_clean:
        return "Available models inquiry"

    # Check for question forms: "How do I ...", "Why does ...", "What is ..."
    q_match = re.match(r"^(?:can you|could you|please|how (?:do|can|to)(?:\s+i)?|why (?:is|does|do)|what (?:is|are))\s+(.*)", raw_query, re.I)
    candidate = q_match.group(1) if q_match else raw_query

    # Clean text of markdown, URLs, tags
    cleaned = _clean_text(candidate)

    # Remove leading articles (the, this, my, a, an)
    cleaned = _LEADING_ARTICLES_RE.sub("", cleaned)

    # Remove trailing question marks and punctuation
    cleaned = cleaned.rstrip("?!.,;:")

    # If first word is a lowercase verb/noun, capitalize nicely
    if cleaned:
        words = cleaned.split()
        # Capitalize first word
        words[0] = words[0].capitalize()
        cleaned = " ".join(words)

    # Truncate to 50 characters at word boundary
    if len(cleaned) > 50:
        cleaned = cleaned[:47].rsplit(" ", 1)[0] + "..."

    return cleaned or "Chat session"
