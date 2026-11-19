"""Embeddings providers and the heading-aware chunker.

EMBEDDINGS_PROVIDER:
- onnx (default): BAAI/bge-small-en-v1.5 exported to ONNX (384 dimensions), run with ONNX Runtime on CPU. No torch.
  The model files are baked into the image at build time (Dockerfile, pinned revision + checksums).
- ollama: an Ollama already running in the cluster (OLLAMA_URL, model OLLAMA_EMBED_MODEL): on mk4, bge-m3:567m at
  http://bge-m3.embeddings:11434 (1024 dimensions, no prefixes). Similarity cutoffs follow the model (app/calibration.py).
- api: a third-party OpenAI-compatible /embeddings endpoint (EMBEDDINGS_API_URL + EMBEDDINGS_API_KEY). Off by default.
- hash: deterministic bag-of-words hashing, for tests and local development only.

Chunking (v5, token-aware): a small fence-aware Markdown splitter by h1–h3, then sections are packed paragraph by
paragraph to a TOKEN budget measured with the embedding model's own tokenizer (ONNX: tokenizer.json; other providers:
≈ 4 characters per token). The budget counts the "title + heading path" prefix that is embedded with every chunk:
target ~350 tokens, hard maximum 480 (bge-small truncates at 512 including [CLS]/[SEP]). Paragraph boundaries are never
inside a fenced code block. A code block too big for one chunk stays whole in the stored `content` (display and
citations) but is embedded as several windows (split at blank lines inside the block, ~400 tokens, small overlap), one
row per window with the same heading path and anchor and a `part` index. Oversized prose is split at lines/sentences.
v4's character chunker (1,800 characters, code blocks whole and unbounded) is kept as `chunk_markdown_v4` for the
retrieval evaluation baseline only. Each chunk keeps its heading path
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

TARGET_CHARS = 1800  # v4 chunker (evaluation baseline only)
MAX_CHARS = 3600

MODEL_MAX_TOKENS = 512  # bge-small-en-v1.5 position embeddings, including [CLS] and [SEP]
SPECIAL_TOKENS = 2
TARGET_TOKENS = 350  # per embedded input, prefix included
HARD_MAX_TOKENS = 480  # never exceeded by an embedded input (prefix included); the rest of 512 is slack
CODE_WINDOW_TOKENS = 400
CODE_WINDOW_OVERLAP_LINES = 2
CHUNKER_VERSION = "tok-v1"  # part of the stored hash: changing the chunker re-embeds everything once


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ModelTokenCounter:
    """Exact token count with the model's tokenizer, without special tokens (the model adds 2)."""

    def __init__(self, tokenizer: Any, name: str = "model"):
        self.tok = tokenizer
        self.name = name

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)


class CharTokenCounter:
    """Fallback for providers without a local tokenizer (hash, ollama, api): ≈ 4 characters per token, rounded up.
    Pessimistic for English prose (bge: ~4.3 chars/token) and roughly right for code."""

    name = "chars/4"

    def count(self, text: str) -> int:
        return -(-len(text) // 4)


def token_counter(emb: Any) -> TokenCounter:
    tok = getattr(emb, "counter_tokenizer", None)
    return ModelTokenCounter(tok, getattr(emb, "name", "model")) if tok is not None else CharTokenCounter()


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
        # A second instance without truncation/padding: the chunker counts real lengths with it (TokenBudget).
        self.counter_tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MODEL_MAX_TOKENS)
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


# Per Ollama model (the name before ":tag"): the prefix it was trained to expect on a query and on a passage, and its
# vector size. A model not listed here embeds without a prefix and needs OLLAMA_EMBED_DIM.
OLLAMA_MODELS: dict[str, tuple[str, str, int]] = {
    "nomic-embed-text": ("search_query: ", "search_document: ", 768),
    "bge-m3": ("", "", 1024),  # no instruction: the model card embeds queries and passages alike
}


class OllamaEmbeddings:
    """An Ollama in the cluster (POST /api/embed), e.g. mk4's bge-m3 service at http://bge-m3.embeddings:11434
    (kubernetes repository, deploy/minikube/embeddings/bge-m3).

    Requests carry BATCH texts: on mk4's CPU service bge-m3 embeds about 0.7 chunks a second, so one article's 40 new
    chunks in a single request would outlast any sensible timeout. Every answer is checked for the expected number and
    size of vectors, because the pgvector column type and the news desk's stored vectors depend on the size."""

    BATCH = 8
    # A dropped connection or a 5xx (the pod restarting, the mesh reconnecting) is retried after 2 s, then 8 s. A timeout
    # is not: the batch already had timeout_s, and three of them would hold a reindex for 15 minutes.
    RETRIES = 2

    def __init__(self, url: str, model: str, dim: int = 0, timeout_s: float = 300.0, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport  # tests: httpx.MockTransport
        if not url:
            raise ValueError("EMBEDDINGS_PROVIDER=ollama needs OLLAMA_URL")
        query_prefix, passage_prefix, known = OLLAMA_MODELS.get(model.split(":", 1)[0], ("", "", 0))
        self.url, self.model, self.timeout_s = url.rstrip("/"), model, timeout_s
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.dim = dim or known
        if not self.dim:
            raise ValueError(f"OLLAMA_EMBED_DIM is required for {model!r} (known models: {', '.join(sorted(OLLAMA_MODELS))})")
        self.name = f"ollama:{model}"

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if query else self.passage_prefix
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as c:
            for b in range(0, len(texts), self.BATCH):
                batch = [prefix + t for t in texts[b : b + self.BATCH]]
                vecs = await self._post(c, batch)
                if len(vecs) != len(batch) or any(len(v) != self.dim for v in vecs):
                    raise ValueError(f"{self.name} returned {[len(v) for v in vecs]} for {len(batch)} inputs; expected {self.dim} dimensions each")
                out.extend(_normalize(np.array(vecs, dtype=np.float32)).tolist())
        return out

    async def _post(self, c: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        import asyncio

        for attempt in range(self.RETRIES + 1):
            try:
                r = await c.post(f"{self.url}/api/embed", json={"model": self.model, "input": batch})
                r.raise_for_status()
                return list(r.json().get("embeddings") or [])
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                transient = e.response.status_code >= 500 if isinstance(e, httpx.HTTPStatusError) else not isinstance(e, httpx.TimeoutException)
                if not transient or attempt == self.RETRIES:
                    raise
                await asyncio.sleep(2 * 4**attempt)
        raise AssertionError("unreachable")


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
        return OllamaEmbeddings(settings.ollama_url, settings.ollama_model, getattr(settings, "ollama_embed_dim", 0), getattr(settings, "embeddings_timeout_s", 300.0))
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
    # v5: an oversized code block is one logical chunk embedded as several windows (one row each, same content).
    part: int = 0
    parts: int = 1
    embed_text: str | None = None  # the window that is embedded; None = the content itself
    identifiers: list[str] = field(default_factory=list)
    tokens: int = 0  # tokens of the embedded input (prefix + text), without the 2 special tokens

    @property
    def content_hash(self) -> str:
        win = "" if self.embed_text is None else f"\n{self.part}\n{self.embed_text}"
        return hashlib.sha256(f"{CHUNKER_VERSION}\n{self.heading_path}\n{self.content}{win}".encode()).hexdigest()

    @property
    def text_for_embedding(self) -> str:
        return self.content if self.embed_text is None else self.embed_text


def embed_input(title: str, chunk: Chunk) -> str:
    """Exactly what is embedded for a chunk: title + heading path carry most of the topical signal."""
    return f"{title}\n{chunk.heading_path}\n\n{chunk.text_for_embedding}"


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


def chunk_markdown_v4(body: str, title: str = "") -> list[Chunk]:
    """v4's character chunker, kept only as the retrieval-evaluation baseline (scripts/rag_eval.py)."""
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


def _is_code(block: str) -> bool:
    return bool(_FENCE.match(block.lstrip()))


def _split_hard(text: str, counter: TokenCounter, limit: int) -> list[str]:
    """Cut text into pieces of ≤ limit tokens: at lines, then sentences, then words, then characters."""
    if counter.count(text) <= limit:
        return [text]
    for sep in ("\n", ". ", " "):
        units = text.split(sep)
        if len(units) == 1:
            continue
        out: list[str] = []
        cur = ""
        for u in units:
            cand = f"{cur}{sep}{u}" if cur else u
            if counter.count(cand) <= limit:
                cur = cand
                continue
            if cur:
                out.append(cur)
            if counter.count(u) <= limit:
                cur = u
            else:
                out.extend(_split_hard(u, counter, limit))
                cur = ""
        if cur:
            out.append(cur)
        return out
    # one unbroken token run (a long hash or URL): characters, shrinking until it fits
    n = max(1, len(text) * limit // max(1, counter.count(text)))
    out = []
    i = 0
    while i < len(text):
        piece = text[i : i + n]
        while counter.count(piece) > limit and len(piece) > 1:
            piece = piece[: max(1, len(piece) * 9 // 10)]
        out.append(piece)
        i += len(piece)
    return out


def code_windows(block: str, counter: TokenCounter, limit: int, window: int = CODE_WINDOW_TOKENS, overlap_lines: int = CODE_WINDOW_OVERLAP_LINES) -> list[str]:
    """Windows of an oversized code block for embedding: split at blank lines inside the block, packed to ~`window`
    tokens, each new window repeating the last `overlap_lines` lines of the previous one; nothing exceeds `limit`."""
    window = min(window, limit)
    segments = [seg for seg in re.split(r"\n[ \t]*\n", block) if seg.strip()]
    pieces: list[str] = []
    for seg in segments:
        pieces.extend(_split_hard(seg, counter, window))
    out: list[str] = []
    cur = ""
    for piece in pieces:
        cand = f"{cur}\n\n{piece}" if cur else piece
        if cur and counter.count(cand) > window:
            out.append(cur)
            tail = "\n".join(cur.split("\n")[-overlap_lines:]) if overlap_lines else ""
            cand = f"{tail}\n{piece}" if tail else piece
            if counter.count(cand) > limit:
                cand = piece
        cur = cand
    if cur:
        out.append(cur)
    return [w for w in out if w.strip()]


def chunk_markdown(body: str, title: str = "", counter: TokenCounter | None = None, *, target: int = TARGET_TOKENS, hard_max: int = HARD_MAX_TOKENS) -> list[Chunk]:
    """Token-budgeted chunks (module docstring). `counter` = the embedding model's tokenizer (token_counter(emb));
    without one, ≈ 4 characters per token. Every chunk's embedded input (embed_input) is ≤ hard_max tokens."""
    counter = counter or CharTokenCounter()
    out: list[Chunk] = []
    for path, anchor, text in _sections(body):
        heading = " › ".join(path) or title
        prefix = counter.count(f"{title}\n{heading}\n\n")
        # A pathological title/heading must not starve the text: keep at least 64 tokens for it.
        room_target = max(64, target - prefix)
        room_hard = max(64, hard_max - prefix)
        groups: list[tuple[str, list[str] | None]] = []  # (content, windows or None)
        cur = ""
        for para in _paragraphs_outside_code(text):
            cand = f"{cur}\n\n{para}" if cur else para
            if counter.count(cand) <= room_target:
                cur = cand
                continue
            if cur:
                groups.append((cur, None))
                cur = ""
            if counter.count(para) <= room_target:
                cur = para
            elif _is_code(para) or "```" in para or "~~~" in para:
                if counter.count(para) <= room_hard:
                    groups.append((para, None))  # one block between target and hard max: alone, whole
                else:
                    groups.append((para, code_windows(para, counter, room_hard)))
            else:
                for piece in _split_hard(para, counter, room_target):
                    groups.append((piece, None))
        if cur:
            groups.append((cur, None))
        for content, windows in groups:
            ids = extract_identifiers(content)
            if windows is None:
                out.append(Chunk(index=len(out), heading_path=heading, anchor=anchor, content=content, identifiers=ids))
            else:
                for i, w in enumerate(windows):
                    out.append(Chunk(index=len(out), heading_path=heading, anchor=anchor, content=content, part=i, parts=len(windows), embed_text=w, identifiers=ids))
    for ch in out:
        ch.tokens = counter.count(embed_input(title, ch))
    return out


# --- Exact identifiers (v5 hybrid retrieval) ------------------------------------------------------------------------

CLI_TOOLS = (
    "kubectl", "helm", "terraform", "docker", "git", "istioctl", "aws", "gh", "npm", "npx", "uv", "pip", "psql", "valkey-cli",
    "redis-cli", "minikube", "argocd", "flux", "velero", "cilium", "curl", "systemctl", "journalctl", "kustomize", "eksctl",
    "openssl", "dig", "ssh", "python", "pytest", "go", "tsc", "node", "deck", "promtool", "kind", "k9s", "jq",
)
K8S_KINDS = (
    "StatefulSet", "DaemonSet", "ReplicaSet", "ReplicationController", "PodDisruptionBudget", "HorizontalPodAutoscaler",
    "ScaledObject", "VirtualService", "DestinationRule", "PeerAuthentication", "AuthorizationPolicy", "ServiceAccount",
    "ClusterRole", "ClusterRoleBinding", "RoleBinding", "ConfigMap", "NetworkPolicy", "PersistentVolumeClaim",
    "PersistentVolume", "StorageClass", "CronJob", "CustomResourceDefinition", "ClusterIssuer", "HTTPRoute",
    "NodePool", "EC2NodeClass", "ServiceEntry", "EnvoyFilter", "PodMonitor", "ServiceMonitor", "PrometheusRule",
    "ClusterPolicy", "IngressClass", "ResourceQuota", "LimitRange", "PriorityClass", "RuntimeClass", "ValidatingAdmissionPolicy",
)
_CLI = re.compile(rf"(?<![\w-])({'|'.join(re.escape(t) for t in CLI_TOOLS)})\s+([a-z][a-z0-9-]{{1,30}})(?![\w-])")
_CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
_VERSION = re.compile(r"(?<![\w.])v?\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.]+)?(?![\w.])")
_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]{1,40}(?![\w-])")
_SET = re.compile(r"\bset\s+-[a-z]+(?:\s+[a-z]+)?\b")
_ENV = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_KIND = re.compile(rf"\b({'|'.join(K8S_KINDS)})\b", re.I)
_CAMEL = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")
_INLINE = re.compile(r"`([^`\n]{2,60})`")
_DISTINCT = re.compile(r"[\d_.:/=-]|[a-z][A-Z]|\s")  # a backticked token counts only if it isn't a plain word


def norm_identifier(v: str) -> str:
    """Lower case, quotes/colons/equals and surrounding punctuation dropped, whitespace collapsed:
    `versioning-strategy: increase` and "versioning-strategy increase" compare equal."""
    v = re.sub(r"[\"'`:=,;()]", " ", v.lower())
    return " ".join(v.split()).rstrip(".?!/")


def question_identifiers(question: str, max_n: int = 3) -> list[str]:
    """Candidate exact identifiers in a reader's question: the extracted ones plus every 1–3-word n-gram of the
    normalised question (readers don't write backticks). Chunk identifiers are already distinctive, so a plain-word
    n-gram can only match a distinctive chunk identifier that happens to be spelled that way."""
    out = extract_identifiers(question, limit=40)
    words = [w for w in norm_identifier(question).split() if w]
    words = [w.strip("?!.") for w in words]
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            g = " ".join(words[i : i + n])
            if g and g not in out:
                out.append(g)
    return out[:120]


def extract_identifiers(text: str, limit: int = 40) -> list[str]:
    """Exact tokens worth matching literally (lower-cased): CLI command + subcommand ("kubectl rollout"), long flags,
    `set -o pipefail`-style options, CVE ids, version strings, env var names, multi-word Kubernetes kinds, camelCase
    config keys, and distinctive `inline code`. Plain words are left to full-text search."""
    found: list[str] = []

    def add(v: str) -> None:
        v = norm_identifier(v)
        if 2 <= len(v) <= 60 and v not in found:
            found.append(v)

    for m in _CLI.finditer(text):
        add(f"{m.group(1)} {m.group(2)}")
    for rx in (_CVE, _FLAG, _SET, _ENV, _CAMEL):
        for m in rx.finditer(text):
            add(m.group(0))
    for m in _VERSION.finditer(text):
        v = m.group(0)
        if v.startswith("v") or v.count(".") >= 2 or len(v) >= 4:  # "1.5" alone is too common to be distinctive
            add(v)
    for m in _KIND.finditer(text):
        add(m.group(1))
    for m in _INLINE.finditer(text):
        tok = m.group(1).strip()
        if _DISTINCT.search(tok) and len(tok.split()) <= 4:
            add(tok)
    return found[:limit]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
