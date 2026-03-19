import json
import httpx
import sys

URL = "http://localhost:8080/v1/chat/completions"

# 1. No system prompt — use server default
payload = {
    "model": "ai-agent",
    "messages": [
        {"role": "user", "content": "Index file UserService.java giúp tôi"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "index_file",
                "description": "Index a Java file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }
            }
        }
    ],
    "stream": True
}

print(f"Sending request to {URL}...")
try:
    with httpx.stream("POST", URL, json=payload, timeout=60.0) as response:
        response.raise_for_status()
        print("Response stream RAW:")
        for line in response.iter_lines():
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
                    if "tool_calls" in delta:
                        print(f"\nTOOL_CALL: {json.dumps(delta['tool_calls'], indent=2)}")
                except Exception: pass
            
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
