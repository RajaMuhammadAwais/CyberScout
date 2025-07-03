"""
DeepSeek API utility for AI-powered enrichment, summarization, or analysis.
"""

import os
import requests

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/completions"  # Example endpoint

def deepseek_complete(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Call DeepSeek API for text completion/summarization."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set in environment.")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",  # Adjust as needed for your DeepSeek plan
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Adjust parsing based on DeepSeek's actual response format
    return data.get("choices", [{}])[0].get("text", "")
