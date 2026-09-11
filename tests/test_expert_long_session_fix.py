import unittest
import json
from server.schemas import ChatMessage
from server.openai_format import (
    new_turn_messages,
    _is_meaningful_assistant,
    build_continuation_prompt,
)
from deepseek.client import _encode_cid, _decode_cid
from deepseek.sse import parse_sse_events

class TestExpertLongSessionFix(unittest.TestCase):
    def test_meaningful_assistant_detection(self):
        self.assertFalse(_is_meaningful_assistant(ChatMessage(role="assistant", content="")))
        self.assertFalse(_is_meaningful_assistant(ChatMessage(role="assistant", content="   \n")))
        self.assertFalse(_is_meaningful_assistant(ChatMessage(role="assistant", content=None)))
        self.assertTrue(_is_meaningful_assistant(ChatMessage(role="assistant", content="Real content")))
        self.assertTrue(_is_meaningful_assistant(ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "bash"}}])))

    def test_new_turn_messages_skips_phantom_assistants(self):
        msgs = [
            ChatMessage(role="user", content="Execute tool"),
            ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "bash"}}]),
            ChatMessage(role="tool", content="Tool success", tool_call_id="c1"),
            ChatMessage(role="assistant", content=""),  # Phantom from empty stop
            ChatMessage(role="user", content="Continue"),
            ChatMessage(role="assistant", content=""),  # Another phantom
            ChatMessage(role="user", content="Continue 2"),
        ]
        new = new_turn_messages(msgs)
        roles = [m.role for m in new]
        contents = [m.content for m in new]
        self.assertEqual(roles, ["tool", "user", "user"])
        self.assertEqual(contents, ["Tool success", "Continue", "Continue 2"])

    def test_empty_turn_error_guard(self):
        from server.api import _TurnStream
        import threading

        class MockEmptyUpstream:
            conversation_id = "test:100"
            def events(self):
                return iter([])

        ts = _TurnStream(MockEmptyUpstream(), ["bash"], "deepseek-expert", threading.Lock())
        events = list(ts.events())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "error")
        self.assertIn("without emitting any content", events[0][1])

if __name__ == "__main__":
    unittest.main()
