from __future__ import annotations

from typing import Any

from mem0.embeddings.base import EmbeddingBase
from mem0.embeddings.ollama import OllamaEmbedding
from mem0.llms.ollama import OllamaLLM
from ollama import Client

from ..local_provider import normalize_local_provider_base_url


OLLAMA_MEMORY_TIMEOUT_SECONDS = 120.0
_CPU_ONLY_EMBED_OPTIONS = {"num_gpu": 0}


def secure_ollama_client(base_url: str) -> Client:
    """Build a bounded Ollama client that stays on the validated local origin."""

    return Client(
        host=normalize_local_provider_base_url(base_url),
        follow_redirects=False,
        trust_env=False,
        timeout=OLLAMA_MEMORY_TIMEOUT_SECONDS,
    )


class AtlasOllamaEmbedding(OllamaEmbedding):
    """Mem0 Ollama embedder with a safe client before its first request."""

    def __init__(self, config: Any = None):
        EmbeddingBase.__init__(self, config)
        self.config.model = self.config.model or "nomic-embed-text"
        self.config.embedding_dims = self.config.embedding_dims or 512
        self.client = secure_ollama_client(self.config.ollama_base_url)
        self._ensure_model_exists()

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """Embed one memory query without competing with the chat model for VRAM."""

        response = self.client.embed(
            model=self.config.model,
            input=text,
            options=_CPU_ONLY_EMBED_OPTIONS,
            keep_alive=0,
        )
        embeddings = response.get("embeddings") or []
        if not embeddings:
            raise ValueError(
                f"Ollama embed() returned no embeddings for model '{self.config.model}'"
            )
        return embeddings[0]

    def embed_batch(
        self,
        texts: list[str],
        memory_action: str = "add",
    ) -> list[list[float]]:
        """Embed a memory batch on CPU and release the runner after the request."""

        if not texts:
            return []
        response = self.client.embed(
            model=self.config.model,
            input=texts,
            options=_CPU_ONLY_EMBED_OPTIONS,
            keep_alive=0,
        )
        embeddings = response.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama embed() returned {len(embeddings)} embeddings for "
                f"{len(texts)} texts using model '{self.config.model}'"
            )
        return embeddings


class AtlasOllamaLLM(OllamaLLM):
    """Mem0 Ollama LLM whose transport stays on the local origin."""

    def __init__(self, config: Any = None):
        # Mem0's parent only constructs its client here; it performs no request.
        # Reuse its config conversion, close that client, and replace it before
        # any model operation can occur.
        super().__init__(config)
        inherited_client = self.client
        close = getattr(inherited_client, "close", None)
        if callable(close):
            close()
        self.client = secure_ollama_client(self.config.ollama_base_url)
