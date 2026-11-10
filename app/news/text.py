"""Deterministic text utilities: URL canonicalisation, cleaning, language, entities, SimHash/MinHash shingles, n-grams,
readability. No network, no model: everything here is unit-tested on fixtures (tests/test_news_text.py)."""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# --- URLs -----------------------------------------------------------------------------------------------------------

TRACKING = re.compile(r"^(utm_[a-z]+|gclid|fbclid|mc_cid|mc_eid|msclkid|yclid|igshid|ref|ref_src|source|cmpid|campaign|_hsenc|_hsmi|spm|trk|sc_channel|sc_campaign|sc_publisher|sc_country|sc_geo|sc_outcome)$", re.I)


def normalize_url(url: str) -> str:
    """Canonical form used for dedupe: https preferred as given, lower-case host without default port or `www.`,
    tracking parameters and fragments removed, remaining query sorted, trailing slash trimmed (except the root)."""
    p = urlsplit(url.strip())
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    port = p.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if not TRACKING.match(k)))
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:32]


def host_of(url: str) -> str:
    h = (urlsplit(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


# --- Cleaning -------------------------------------------------------------------------------------------------------

_DROP = re.compile(r"<(script|style|noscript|svg|nav|footer|header|form|iframe|aside)\b[\s\S]*?</\1>", re.I)
_BLOCK = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article|pre|blockquote)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")


def clean_text(raw: str, limit: int = 20_000) -> str:
    """HTML or text → readable plain text (paragraphs kept), entities decoded, bounded."""
    t = _DROP.sub(" ", raw or "")
    t = _BLOCK.sub("\n", t)
    t = _TAG.sub(" ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t\r\f\v ]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:limit]


def content_hash(title: str, text: str) -> str:
    norm = re.sub(r"\W+", " ", f"{title} {text}".lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()[:32]


# --- Language -------------------------------------------------------------------------------------------------------

_STOP = {
    "en": {"the", "and", "of", "to", "in", "is", "for", "that", "with", "on", "this", "are", "as", "be", "by", "it", "from", "an", "or", "you", "we", "can", "new", "now"},
    "de": {"der", "die", "und", "das", "ist", "nicht", "mit", "den", "von", "zu", "ein", "eine", "auf", "für", "sich"},
    "fr": {"le", "la", "les", "et", "des", "est", "une", "un", "pour", "dans", "que", "sur", "pas", "avec", "du"},
    "es": {"el", "la", "los", "las", "y", "de", "que", "en", "es", "una", "un", "para", "con", "por", "del"},
}


def detect_language(text: str) -> str | None:
    """Stop-word vote over the first ~400 words. Returns None when there is too little text to tell."""
    words = re.findall(r"[a-zäöüßéèàçñ]+", text.lower())[:400]
    if len(words) < 8:
        return None
    scores = {lang: sum(1 for w in words if w in stop) for lang, stop in _STOP.items()}
    lang, best = max(scores.items(), key=lambda kv: kv[1])
    return lang if best >= max(2, len(words) * 0.04) else None


# --- Entities (rules + gazetteer; docs/news/DECISIONS.md N-14) ------------------------------------------------------

CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
VERSION = re.compile(r"(?<![\w.])v?(\d+\.\d+(?:\.\d+)?(?:-(?:alpha|beta|rc|preview)\.?\d*)?)(?![\w]|\.\d)", re.I)
GAZETTEER: dict[str, tuple[str, ...]] = {
    "product": (
        "Kubernetes", "Terraform", "OpenTofu", "Argo CD", "Argo Workflows", "Istio", "Kong", "Prometheus", "Grafana", "Loki", "Tempo", "Jaeger",
        "OpenTelemetry", "Helm", "Kustomize", "Docker", "containerd", "Podman", "Envoy", "Cilium", "Calico", "Flux", "Crossplane", "Karpenter",
        "EKS", "AKS", "GKE", "Lambda", "Bedrock", "SageMaker", "S3", "EC2", "CloudFront", "IAM", "Azure OpenAI", "Vertex AI", "Gemini",
        "Claude", "GPT", "LangChain", "LangGraph", "CrewAI", "MCP", "Model Context Protocol", "Ollama", "vLLM", "Ray", "MLflow", "Kubeflow",
        "PostgreSQL", "pgvector", "Redis", "Kafka", "Vault", "Consul", "Nomad", "Packer", "GitHub Actions", "GitLab", "Jenkins", "Backstage",
        "cert-manager", "Trivy", "Falco", "Kyverno", "OPA", "Gatekeeper", "Sigstore", "cosign", "Harbor", "Minikube", "kind", "k3s",
    ),
    "org": (
        "AWS", "Amazon", "Microsoft", "Azure", "Google", "Google Cloud", "GCP", "CNCF", "Linux Foundation", "HashiCorp", "IBM", "Red Hat",
        "Anthropic", "OpenAI", "Meta", "Mistral", "Hugging Face", "GitHub", "GitLab", "Docker Inc", "CISA", "NIST", "NVD", "OWASP",
        "Datadog", "Cloudflare", "Oracle", "VMware", "Broadcom", "SUSE", "Canonical",
    ),
}
_GAZ = [(kind, name, re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.I if len(name) > 4 else 0)) for kind, names in GAZETTEER.items() for name in names]


def extract_entities(title: str, text: str) -> dict[str, list[str]]:
    """Products, organisations, CVE ids and version strings mentioned in the item. Capped lists, stable order."""
    body = f"{title}\n{text[:12_000]}"
    out: dict[str, list[str]] = {"product": [], "org": [], "cve": [], "version": []}
    for kind, name, rx in _GAZ:
        if rx.search(body) and name not in out[kind]:
            out[kind].append(name)
    out["cve"] = sorted({m.upper() for m in CVE.findall(body)})[:20]
    out["version"] = list(dict.fromkeys(m for m in VERSION.findall(body)))[:20]
    for k in ("product", "org"):
        out[k] = out[k][:20]
    return out


def entity_set(entities: dict[str, list[str]]) -> set[str]:
    return {f"{k}:{v.lower()}" for k, vs in entities.items() if k != "version" for v in vs}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Shingles, SimHash, n-grams -------------------------------------------------------------------------------------

WORD = re.compile(r"[a-z0-9][a-z0-9'.+#-]*", re.I)


def words(text: str) -> list[str]:
    return [w.lower().strip(".'") for w in WORD.findall(text) if w.strip(".'")]


def shingles(text: str, k: int = 5) -> set[str]:
    w = words(text)
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def simhash(text: str) -> int:
    """64-bit SimHash over word 3-grams, returned as a signed int64 (fits Postgres bigint)."""
    v = [0] * 64
    for sh in shingles(text, 3):
        h = _h64(sh)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= 1 << i
    return out - (1 << 64) if out >= (1 << 63) else out


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


# --- Readability ----------------------------------------------------------------------------------------------------


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def sentences(text: str) -> list[str]:
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"`[^`]*`", "code", t)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n{2,}|\n(?=[-*] |\d+\. |#)", t)
    return [p.strip(" -*#\n") for p in parts if len(p.strip(" -*#\n")) > 2]


def flesch_reading_ease(text: str) -> float:
    ss = sentences(text)
    ws = [w for w in re.findall(r"[A-Za-z]+", re.sub(r"```[\s\S]*?```", " ", text))]
    if not ss or not ws:
        return 0.0
    syl = sum(_syllables(w) for w in ws)
    return round(206.835 - 1.015 * (len(ws) / len(ss)) - 84.6 * (syl / len(ws)), 1)


def top_terms(text: str, n: int = 8) -> list[str]:
    stop = _STOP["en"] | {"will", "has", "have", "was", "were", "its", "their", "which", "also", "more", "than", "into", "about", "your", "our", "not", "but", "all", "how"}
    c = Counter(w for w in words(text) if len(w) > 2 and w not in stop and not w.isdigit())
    return [w for w, _ in c.most_common(n)]
