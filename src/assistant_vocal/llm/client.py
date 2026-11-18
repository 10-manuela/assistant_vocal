"""Client Ollama en streaming (bibliothèque standard uniquement)."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from assistant_vocal.config import AppConfig


class LLMError(RuntimeError):
    """Erreur de communication avec Ollama."""


@dataclass
class ChatStats:
    first_token_s: float | None  # délai avant le premier mot
    total_s: float
    tokens: int
    tokens_per_s: float


def make_message(role: str, content: str, images: list[bytes] | None = None) -> dict:
    """Construit un message. Les images (octets PNG/JPEG) sont encodées en base64."""
    msg = {"role": role, "content": content}
    if images:
        msg["images"] = [base64.b64encode(i).decode("ascii") for i in images]
    return msg


class LLMClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        timeout: float = 120.0,
        opener=None,  # injectable pour les tests
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen
        self.last_stats: ChatStats | None = None
        self.last_tool_calls: list[dict] = []

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "LLMClient":
        c = cfg.llm
        return cls(c.host, c.model, c.temperature, c.num_ctx, c.keep_alive, c.timeout)

    def _request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def warmup(self) -> None:
        """Charge le modèle en mémoire (sans générer de texte)."""
        payload = {"model": self.model, "messages": [], "keep_alive": self.keep_alive}
        try:
            with self._open(self._request(payload), timeout=self.timeout) as resp:
                resp.read() if hasattr(resp, "read") else list(resp)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except OSError as exc:
            raise LLMError(f"Ollama injoignable ({exc})") from exc

    def _http_error(self, exc: urllib.error.HTTPError) -> LLMError:
        if exc.code == 404:
            return LLMError(f"Modèle '{self.model}' absent. Lance : ollama pull {self.model}")
        return LLMError(f"Erreur HTTP {exc.code} d'Ollama")

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[str]:
        """Génère la réponse morceau par morceau.

        Fermer le générateur (break, .close()) coupe la requête : c'est ce qui
        permettra l'interruption quand l'utilisateur coupe la parole.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if tools:
            payload["tools"] = tools

        self.last_tool_calls = []
        self.last_stats = None
        start = time.perf_counter()
        first: float | None = None
        tokens = 0
        eval_seconds: float | None = None

        try:
            with self._open(self._request(payload), timeout=self.timeout) as resp:
                for raw in resp:
                    raw = raw.strip()
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if "error" in data:
                        raise LLMError(data["error"])
                    msg = data.get("message") or {}
                    if msg.get("tool_calls"):
                        self.last_tool_calls.extend(msg["tool_calls"])
                    piece = msg.get("content", "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - start
                        yield piece
                    if data.get("done"):
                        tokens = data.get("eval_count", 0)
                        if data.get("eval_duration"):
                            eval_seconds = data["eval_duration"] / 1e9
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except OSError as exc:  # inclut URLError, ConnectionError, TimeoutError
            raise LLMError(f"Ollama injoignable ({exc})") from exc

        total = time.perf_counter() - start
        speed = tokens / eval_seconds if tokens and eval_seconds else 0.0
        self.last_stats = ChatStats(first, total, tokens, speed)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return "".join(self.stream(messages, tools))