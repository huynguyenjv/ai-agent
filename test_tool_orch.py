"""Standalone test for tool_orchestrator — no venv dependencies needed."""
import json, re


# ── Inline copy of the core logic (avoids import chain → jinja2/langchain) ──

PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

def parse_tool_calls(text):
    calls = []
    for match in PATTERN.finditer(text):
        content = match.group(1).strip()
        try:
            call_data = json.loads(content)
            if "name" in call_data:
                calls.append(call_data)
        except json.JSONDecodeError:
            pass
    return calls

def extract_partial(text):
    start_tag, end_tag = "<tool_call>", "</tool_call>"
    idx = text.rfind(start_tag)
    if idx == -1:
        return None
    if text.find(end_tag, idx) != -1:
        return None
    return text[idx + len(start_tag):]


# ── Streaming tag detection simulation ──
def simulate_stream(chunks: list[str], debug=False):
    """Simulate the _stream_from_vllm tag detection logic."""
    START_TAG = "<tool_call>"
    END_TAG = "</tool_call>"
    TAG_MAX_LEN = len(START_TAG)

    in_tool_tag = False
    tool_buffer = ""
    pending_text = ""
    yielded_content = []
    tool_calls_found = []

    for item in chunks:
        if in_tool_tag:
            tool_buffer += item
            if END_TAG in tool_buffer:
                raw = tool_buffer.split(END_TAG)[0].strip()
                remainder = tool_buffer.split(END_TAG, 1)[1]
                if "{" in raw:
                    raw = raw[raw.find("{"):]
                try:
                    tool_calls_found.append(json.loads(raw))
                except:
                    pass
                in_tool_tag = False
                tool_buffer = ""
                if remainder.strip():
                    pending_text = remainder
            continue

        pending_text += item

        if START_TAG in pending_text:
            before, after = pending_text.split(START_TAG, 1)
            if before:
                yielded_content.append(before)
            pending_text = ""
            # Check if the after portion already contains the end tag
            if END_TAG in after:
                raw = after.split(END_TAG)[0].strip()
                remainder = after.split(END_TAG, 1)[1]
                if "{" in raw:
                    raw = raw[raw.find("{"):]
                try:
                    tool_calls_found.append(json.loads(raw))
                except:
                    pass
                if remainder.strip():
                    pending_text = remainder
            else:
                in_tool_tag = True
                tool_buffer = after
            continue

        might_be_tag = any(pending_text.endswith(START_TAG[:i]) for i in range(1, min(TAG_MAX_LEN, len(pending_text) + 1)))
        if might_be_tag:
            safe_end = len(pending_text) - TAG_MAX_LEN
            if safe_end > 0:
                yielded_content.append(pending_text[:safe_end])
                pending_text = pending_text[safe_end:]
        else:
            yielded_content.append(pending_text)
            pending_text = ""

    if pending_text and not in_tool_tag:
        yielded_content.append(pending_text)

    return "".join(yielded_content), tool_calls_found


# ── Tests ──

def test_parse():
    t1 = 'Hello\n<tool_call>\n{"name": "idx", "arguments": {"f": "A.java"}}\n</tool_call>'
    c1 = parse_tool_calls(t1)
    assert len(c1) == 1 and c1[0]["name"] == "idx"

    t2 = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>'
    assert len(parse_tool_calls(t2)) == 2

    assert len(parse_tool_calls("no tools")) == 0
    assert len(parse_tool_calls('<tool_call>bad json</tool_call>')) == 0
    print("✅ parse_tool_calls: all cases pass")


def test_partial():
    assert extract_partial('text <tool_call>{"name"') is not None
    assert extract_partial('<tool_call>{}</tool_call>') is None
    assert extract_partial("no tags") is None
    print("✅ extract_partial: all cases pass")


def test_stream_normal():
    """Normal text — no tool tags."""
    text, calls = simulate_stream(["Hello ", "world!"])
    assert text == "Hello world!" and len(calls) == 0
    print("✅ stream: normal text passes through")


def test_stream_single_tool():
    """Single tool call in stream."""
    chunks = ["Let me do that. ", '<tool_call>\n{"name": "idx"', ', "arguments": {}}\n</tool_call>', " Done!"]
    text, calls = simulate_stream(chunks)
    assert "Let me do that." in text, f"Pre-tag text missing: '{text}'"
    assert len(calls) == 1 and calls[0]["name"] == "idx"
    print("✅ stream: single tool call detected, pre-tag text preserved")


def test_stream_split_tag():
    """Tag split across chunks: '<tool_' + 'call>\n..."""
    chunks = ["Text before ", "<tool_", 'call>\n{"name": "x", "arguments": {}}\n</tool_call>']
    text, calls = simulate_stream(chunks, debug=True)
    assert len(calls) == 1 and calls[0]["name"] == "x", f"Expected 1 call, got {len(calls)}: {calls}\ntext='{text}'"
    print(f"✅ stream: split tag detected correctly (text='{text.strip()}', calls={len(calls)})")


def test_stream_multi_tool():
    """Multiple tool calls."""
    chunks = [
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>',
        " between ",
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>',
    ]
    text, calls = simulate_stream(chunks)
    assert len(calls) == 2 and calls[0]["name"] == "a" and calls[1]["name"] == "b"
    print("✅ stream: multiple tool calls detected")


if __name__ == "__main__":
    print("=== Tool Orchestration Tests ===\n")
    test_parse()
    test_partial()
    test_stream_normal()
    test_stream_single_tool()
    test_stream_split_tag()
    test_stream_multi_tool()
    print("\n=== All tests passed ✅ ===")
