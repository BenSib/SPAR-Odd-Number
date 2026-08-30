import sys
import requests

API_URL = "http://localhost:11434/v1/chat/completions"
MAX_TOKENS = 20

# Models registered in Ollama, pointing at the GGUF files under
# C:\Users\benws\Models. Base models have no chat template (they're raw
# completion models), so their output for a chat-style prompt is less
# reliable than the instruct variants.
MODELS = {
    "llama3.1-8b-instruct": "llama3.1-8b-instruct-local",
    "llama3.1-8b-base": "llama3.1-8b-base-local",
    "qwen3-8b-instruct": "qwen3-8b-instruct-local",
    "qwen3-8b-base": "qwen3-8b-base-local",
    # Same weights as qwen3-8b-instruct, but without the /no_think override,
    # so the model's chain-of-thought is left intact. Needs a much larger
    # max_tokens (see reward_gaming_experiment.py's --max-tokens) or thinking
    # gets cut off mid-thought.
    "qwen3-8b-instruct-thinking": "qwen3-8b-instruct-thinking-local",
}


def call_local(prompt, model="llama3.1-8b-instruct", max_tokens=MAX_TOKENS):
    model_id = MODELS.get(model, model)

    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    )
    if not response.ok:
        print(f"Status: {response.status_code}", file=sys.stderr)
        print(f"Body: {response.text}", file=sys.stderr)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Choose a number from 1 to 10"
    model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1-8b-instruct"
    print(call_local(prompt, model=model))
