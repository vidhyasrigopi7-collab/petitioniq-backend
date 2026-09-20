import os
import json
import requests
from fastapi import HTTPException
_runtime = {"provider": None, "keys": {}}
_KEY_ENV_NAMES = {
    "claude": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",}
def set_runtime_provider(provider, api_key=None):
    _runtime["provider"] = provider
    env_name = _KEY_ENV_NAMES.get(provider)
    if env_name and api_key:
        _runtime["keys"][env_name] = api_key
def get_runtime_settings():
    """Returns the currently active provider and which keys are known (never
    the key values themselves) — safe to send to the frontend."""
    provider = _runtime["provider"] or os.getenv("LLM_PROVIDER", "ollama")
    configured = {
        name: bool(_runtime["keys"].get(env_name) or os.getenv(env_name))
        for name, env_name in _KEY_ENV_NAMES.items()
    }
    return {"provider": provider, "configured": configured}
def _get_key(env_name):
    return _runtime["keys"].get(env_name) or os.getenv(env_name)
def _extract_json(text):
    """Best-effort JSON parse: try the whole string, then scan for the first
    balanced {...} block (counting nested braces properly, not just the
    first/last brace in the text — reasoning models sometimes wrap the JSON
    in extra text that can contain stray braces of its own)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise HTTPException(
        status_code=502,
        detail="Model output was cut off or malformed. Try again, "
               "or the document may be too long/complex for a single pass."
    )
class LLMProvider:
    """Interface every provider implements."""
    def generate_json(self, prompt, num_predict=1000):
        raise NotImplementedError

    def generate_text(self, prompt, num_predict=500):
        raise NotImplementedError
class OllamaProvider(LLMProvider):
    def __init__(self):
        self.url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        # Local models (especially on CPU-only machines) can take a long time
        # for long outputs like a full counter-petition draft — default to a
        # generous timeout, overridable via OLLAMA_TIMEOUT (seconds).
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "900"))
    def _chat(self, prompt, num_predict, json_mode):
        try:
            cpu_threads = int(os.getenv("OLLAMA_NUM_THREAD", str(min(os.cpu_count() or 4, 8))))
            ctx_size = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
            predict_limit = min(num_predict, 2500) if not json_mode else min(num_predict, 1500)
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0 if json_mode else 0.1,
                    "num_ctx": ctx_size,
                    "num_predict": predict_limit,
                    "num_thread": cpu_threads,
                    "top_k": 20,
                    "top_p": 0.9,
                },
            }
            if json_mode:
                body["format"] = "json"
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
            if r.status_code != 200:
                raise Exception(f"Ollama HTTP {r.status_code}")
            return r.json()["message"]["content"]
        except requests.Timeout:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Ollama timed out after {self.timeout}s. Long drafts (like a counter "
                    "petition) take longer on local/CPU models — you can raise the "
                    "OLLAMA_TIMEOUT environment variable, or try a smaller/faster model."
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    def generate_json(self, prompt, num_predict=1000):
        return _extract_json(self._chat(prompt, num_predict, json_mode=True))

    def generate_text(self, prompt, num_predict=500):
        return self._chat(prompt, num_predict, json_mode=False).strip()
class ClaudeProvider(LLMProvider):
    def __init__(self):
        self.api_key = _get_key("ANTHROPIC_API_KEY")
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        if not self.api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set.")

    def _call(self, prompt, num_predict):
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": num_predict,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            if r.status_code != 200:
                raise Exception(f"Claude HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            return "".join(block.get("text", "") for block in data.get("content", []))
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Claude timed out.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Claude request failed: {e}")

    def generate_json(self, prompt, num_predict=1000):
        content = self._call(prompt + "\n\nRespond with ONLY valid JSON, no other text.", num_predict)
        return _extract_json(content)

    def generate_text(self, prompt, num_predict=500):
        return self._call(prompt, num_predict).strip()
class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = _get_key("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        if not self.api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")

    def _call(self, prompt, num_predict):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": num_predict,
                    "temperature": 0.2,
                },
                timeout=120,
            )
            if r.status_code != 200:
                if r.status_code in (413, 429):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Groq rejected this request — the document/prompt is bigger than "
                            "your Groq account's current tokens-per-minute limit allows. Try "
                            "again in a minute, use a shorter document, or switch to Ollama "
                            "(local) in Settings for large petitions/law books."
                        ),
                    )
                raise Exception(f"Groq HTTP {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"]
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Groq timed out.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq request failed: {e}")

    def generate_json(self, prompt, num_predict=1000):
        content = self._call(prompt + "\n\nRespond with ONLY valid JSON, no other text.", num_predict)
        return _extract_json(content)

    def generate_text(self, prompt, num_predict=500):
        return self._call(prompt, num_predict).strip()
class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = _get_key("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not self.api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")

    def _call(self, prompt, num_predict):
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": num_predict, "temperature": 0.2},
                },
                timeout=120,
            )
            if r.status_code != 200:
                raise Exception(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Gemini timed out.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini request failed: {e}")

    def generate_json(self, prompt, num_predict=1000):
        content = self._call(prompt + "\n\nRespond with ONLY valid JSON, no other text.", num_predict)
        return _extract_json(content)

    def generate_text(self, prompt, num_predict=500):
        return self._call(prompt, num_predict).strip()
def get_llm_provider():
    """Returns the active provider — checks the runtime Settings override first (set via the app's Settings panel), then falls back to the LLM_PROVIDER environment variable, then to "ollama" if neither is set. Supported values: "ollama", "claude", "groq", "gemini"."""
    name = (_runtime["provider"] or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if name == "claude":
        return ClaudeProvider()
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    return OllamaProvider()