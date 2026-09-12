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

    def test_sse_context_length_exceeded_frame(self):
        raw_lines = [
            'event: ready',
            'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}',
            'event: hint',
            'data: {"type":"error","content":"Length limit reached. Please start a new chat.","clear_response":true,"finish_reason":"context_length_exceeded"}',
            'event: close',
            'data: {"click_behavior":"none","auto_resume":false}',
        ]
        meta = {}
        events = list(parse_sse_events(raw_lines, meta))
        self.assertEqual(meta.get("error"), "context_length_exceeded")
        self.assertIn("Length limit reached", meta.get("error_msg", ""))

    def test_invalid_conversation_markers_context_length(self):
        from server.api import _looks_like_invalid_conversation
        self.assertTrue(_looks_like_invalid_conversation("Context length limit reached on upstream session"))
        self.assertTrue(_looks_like_invalid_conversation("Length limit reached. Please start a new chat."))
        self.assertTrue(_looks_like_invalid_conversation("finish_reason: context_length_exceeded"))
        self.assertFalse(_looks_like_invalid_conversation("Connection reset by peer"))

    def test_dsml_write_with_nested_invoke_text(self):
        from server.tools_bridge import parse_tool_calls

        sample_resp = '''I will write the document now.

<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="write">
<｜｜DSML｜｜ parameter name="content" string="true"># Architecture
Here is an explanation mentioning <invoke> tags inside text.
</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="filePath" string="true">/path/to/doc.md</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''

        tools_def = [
            {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}}
        ]

        calls, clean = parse_tool_calls(sample_resp, tools_def)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "write")
        self.assertEqual(calls[0].arguments.get("filePath"), "/path/to/doc.md")
        self.assertIn("Here is an explanation", calls[0].arguments.get("content", ""))

    def test_benign_text_with_tool_keywords_emitted_as_content(self):
        from server.api import _TurnStream
        import threading

        sample_markdown = 'In `tools_bridge.py`, we match `<tool>` and `DSML` patterns.'

        class MockUpstreamText:
            conversation_id = "test:101"
            def events(self):
                yield ("content", sample_markdown)

        ts = _TurnStream(MockUpstreamText(), ["bash", "read"], "deepseek-expert", threading.Lock())
        events = list(ts.events())
        self.assertEqual(ts.finish_reason, "stop")
        content_events = [v for k, v in events if k == "content"]
        self.assertTrue(len(content_events) > 0)
        full_emitted = "".join(content_events)
        self.assertEqual(full_emitted, sample_markdown)

if __name__ == "__main__":
    unittest.main()
