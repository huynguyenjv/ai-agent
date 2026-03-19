import json
import httpx
import sys

# URL of the actual vLLM backend
URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "messages": [
        {"role": "user", "content": "Hi."}
    ],
    "stream": True
}

print(f"Pinging vLLM directly at {URL}...")
try:
    with httpx.stream("POST", URL, json=payload, timeout=30.0) as response:
        response.raise_for_status()
        print("Response stream:")
        for line in response.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                data = json.loads(data_str)
                print(data["choices"][0]["delta"].get("content", ""), end="", flush=True)
except Exception as e:
    print(f"\nDirect Ping Error: {e}")
    sys.exit(1)
