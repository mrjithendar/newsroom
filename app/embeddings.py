"""Embeddings providers and the heading-aware chunker.

EMBEDDINGS_PROVIDER:
- onnx (default): BAAI/bge-small-en-v1.5 exported to ONNX (384 dimensions), run with ONNX Runtime on CPU. No torch.
  The model files are baked into the image at build time (Dockerfile, pinned revision + checksums).
- ollama: an Ollama already running in the cluster (OLLAMA_URL, model OLLAMA_EMBED_MODEL, e.g. nomic-embed-text).
- api: a third-party OpenAI-compatible /embeddings endpoint (EMBEDDINGS_API_URL + EMBEDDINGS_API_KEY). Off by default.
- hash: deterministic bag-of-words hashing, for tests and local development only.

Chunking: a small fence-aware Markdown splitter by h1–h3, then sections longer than the target are cut at paragraph
boundaries that are never inside a fenced code block, so code blocks stay whole. Each chunk keeps its heading path
("Setup › Install") and the anchor of its heading for citations (/blog/slug#anchor). Anchors follow rehype-slug
(github-slugger), including the -1/-2 suffixes for repeated headings, counted over every heading level like the site.
LangChain's MarkdownHeaderTextSplitter was tried first and dropped: it removes blank lines (so paragraph boundaries are
lost) and cannot tell two sections with the same heading apart, which breaks the duplicate-anchor suffixes (D-45).
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

TARGET_CHARS = 1800
MAX_CHARS = 3600


class Embeddings(Protocol):
    name: str
    dim: int

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


class OnnxEmbeddings:
    """bge-small-en-v1.5 via ONNX Runtime: CLS pooling + L2 normalisation (the model card's recipe)."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    BATCH = 4

    def __init__(self, model_dir: str, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.name = f"onnx:{d.name}"
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106 - a token string, not a secret
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # Measured (docs/seo/AGENTS.md "Resources"): with the default memory arena and batches of 16, a full reindex
        # peaked at ~1.15 GiB RSS; without the arena, small length-sorted batches stay a few hundred MiB.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False  # per-shape allocation plans grew RSS across many batch shapes
        self.session = ort.InferenceSession(str(d / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.dim = int(self.session.get_outputs()[0].shape[-1] or 384)

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        out = self.session.run(None, feed)[0]
        return _normalize(out[:, 0, :])

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        import asyncio

        if query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        # Length-sorted, small batches: padding (and attention memory) is bounded by similar-length neighbours.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[list[float] | None] = [None] * len(texts)
        for b in range(0, len(order), self.BATCH):
            idx = order[b : b + self.BATCH]
            vecs = await asyncio.to_thread(self._run, [texts[i] for i in idx])
            for i, v in zip(idx, vecs.tolist(), strict=True):
                out[i] = v
        return [v for v in out if v is not None]


class OllamaEmbeddings:
    def __init__(self, url: str, model: str, dim: int = 768):
        self.url, self.model, self.dim = url, model, dim
        self.name = f"ollama:{model}"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        prefix = "search_query: " if query else "search_document: "
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self.url}/api/embed", json={"model": self.model, "input": [prefix + t for t in texts]})
            r.raise_for_status()
            return _normalize(np.array(r.json()["embeddings"], dtype=np.float32)).tolist()


class ApiEmbeddings:
    def __init__(self, url: str, key: str, model: str = "text-embedding-3-small", dim: int = 1536):
        self.url, self.key, self.model, self.dim = url, key, model, dim
        self.name = f"api:{model}"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, json={"model": self.model, "input": texts})
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]


class HashEmbeddings:
    """Deterministic hashed bag of words (tests/dev). Shares words → similar vectors; not semantic."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash:{dim}"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) < 3:
                    continue
                h = int(hashlib.blake2b(w.encode(), digest_size=8).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            n = float(np.linalg.norm(v))
            out.append((v / n if n else v).tolist())
        return out


def make_embeddings(settings: Any) -> Embeddings:
    p = getattr(settings, "embeddings_provider", "onnx")
    if p == "onnx":
        return OnnxEmbeddings(settings.embeddings_model_dir)
    if p == "ollama":
        return OllamaEmbeddings(settings.ollama_url, settings.ollama_model)
    if p == "api":
        return ApiEmbeddings(settings.embeddings_api_url, settings.embeddings_api_key)
    if p == "hash":
        return HashEmbeddings()
    raise ValueError(f"unknown EMBEDDINGS_PROVIDER {p!r}")


# --- Chunking ------------------------------------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^\w\- ]", re.UNICODE)


def github_slug(text: str) -> str:
    """github-slugger (what rehype-slug uses on the site): lower-case, drop punctuation, spaces → hyphens."""
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"`", "", t)
    return _SLUG_STRIP.sub("", t).replace(" ", "-")


@dataclass
class Chunk:
    index: int
    heading_path: str
    anchor: str | None
    content: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.heading_path}\n{self.content}".encode()).hexdigest()


_FENCE = re.compile(r"^(```|~~~)")


def _paragraphs_outside_code(text: str) -> list[str]:
    """Split on blank lines, but never inside a fenced code block."""
    parts: list[str] = []
    buf: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        m = _FENCE.match(line.strip())
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        if not line.strip() and fence is None:
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


def _visible_heading(text: str) -> str:
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*_`]", "", t).strip()


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def _sections(body: str) -> list[tuple[list[str], str | None, str]]:
    """(heading path, anchor, text) per h1–h3 section, in order. Headings inside code fences are text."""
    seen: dict[str, int] = {}
    stack: list[tuple[int, str]] = []
    out: list[tuple[list[str], str | None, str]] = []
    anchor: str | None = None
    buf: list[str] = []
    fence: str | None = None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            out.append(([t for _, t in stack], anchor, text))
        buf.clear()

    for line in body.split("\n"):
        m = _FENCE.match(line.strip())
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        h = _HEADING.match(line) if fence is None and not m else None
        if h:
            level, text = len(h.group(1)), _visible_heading(h.group(2))
            base = github_slug(text)
            n = seen.get(base, 0)
            seen[base] = n + 1
            slug = base if n == 0 else f"{base}-{n}"
            if level <= 3:
                flush()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                anchor = slug
                continue
        buf.append(line)
    flush()
    return out


def chunk_markdown(body: str, title: str = "") -> list[Chunk]:
    out: list[Chunk] = []
    for path, anchor, text in _sections(body):
        pieces: list[str] = []
        cur = ""
        for para in _paragraphs_outside_code(text):
            if cur and len(cur) + len(para) + 2 > TARGET_CHARS:
                pieces.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            pieces.append(cur)
        for p in pieces:
            # A single code block longer than MAX_CHARS stays whole (never cut mid-code); it is just a long chunk.
            out.append(Chunk(index=len(out), heading_path=" › ".join(path) or title, anchor=anchor, content=p))
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
