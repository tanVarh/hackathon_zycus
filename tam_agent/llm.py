"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        default_models = {
            "gemini": "gemini-3.5-flash-lite",
            "ollama": "llama3.1:8b",
            "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o-mini",
            "mock": "mock-model",
        }
        self.model = model or os.getenv("LLM_MODEL", default_models.get(self.provider, "gemini-3.5-flash-lite"))
        self.calls = 0  # simple counter you can report for cost/latency
        self.last_run_successful_calls = 0
        self.last_run_fallback = False
        self.last_run_fallback_reason = None

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text."""
        self.calls += 1
        if self.provider == "mock":
            return ""
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            contents = []
            for m in messages:
                role = "model" if m.get("role") in ["assistant", "model"] else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.get("content", ""))]))
            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return resp.text or ""
        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                          system=system or "", messages=messages)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                  temperature=temperature)
            return resp.choices[0].message.content or ""
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                               "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
            req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
