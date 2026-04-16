from server.tools.registry import TOOL_SCHEMAS, merge_tools


def test_tool_schemas_has_read_file_and_grep_code():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"read_file", "grep_code"}
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


def test_merge_tools_empty_client_returns_server_schemas():
    merged = merge_tools(None)
    assert len(merged) == len(TOOL_SCHEMAS)
    assert merged == TOOL_SCHEMAS


def test_merge_tools_client_only_tool_preserved():
    client = [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}]
    merged = merge_tools(client)
    names = {t["function"]["name"] for t in merged}
    assert names == {"read_file", "grep_code", "foo"}


def test_merge_tools_server_overrides_on_name_collision():
    fake = {"type": "function", "function": {"name": "read_file", "description": "HACKED", "parameters": {"type": "object", "properties": {}}}}
    merged = merge_tools([fake])
    read_file = next(t for t in merged if t["function"]["name"] == "read_file")
    assert read_file["function"]["description"] != "HACKED"
