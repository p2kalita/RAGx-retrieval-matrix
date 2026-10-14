
import os
from openai import OpenAI

BASE_URL = "https://dried-moonscape-violet.ngrok-free.dev/v1"

# Use the API key configured on your vLLM server.
# If authentication is disabled, a placeholder may work.
API_KEY = os.getenv("VLLM_API_KEY", "local-dev-key")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=120.0,
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in three bullet points."
        }
    ],
    temperature=0.0,
    max_tokens=600,
    seed=42,
)

print(response.choices[0].message.content)
