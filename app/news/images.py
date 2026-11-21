"""Cover images (docs/news/IMAGES.md).

Providers (NEWS_IMAGE_PROVIDER / the image profile; none but `template` works without explicit configuration):
- template (default, free): a generated cover in the site's colours (category tone), a deterministic geometric
  illustration per topic and the headline. Pillow, no network. Always available as the fallback.
- gemini: the web app's existing Gemini image integration (Admin → Integrations, its own key and limits) through the
  callback API; the agent never holds that key.
- openai: the OpenAI Images API called from the agent (needs OPENAI_API_KEY in the news-agent Secret and an egress rule).
Local Stable Diffusion is not offered: on CPU-only Minikube one 1024² image takes minutes and several GiB of RAM.

Pipeline: brief → N variants → vision safety check (no faces, logos/trademarks, text gibberish) → best match to the
brief → optimise (EXIF stripped; WebP 1200×630 OG + 1600×900 hero + 800w/400w) → stored in the CMS media library via
the web app → alt text written by a vision model looking at the *final* image (80–125 characters).
AI-generated images carry IPTC DigitalSourceType trainedAlgorithmicMedia in their XMP and `ai_generated` in news_images.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import random
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

TONES: dict[str, tuple[str, str]] = {
    "violet": ("#4c1d95", "#7c3aed"),
    "fuchsia": ("#701a75", "#c026d3"),
    "indigo": ("#312e81", "#4f46e5"),
    "amber": ("#78350f", "#d97706"),
    "sky": ("#0c4a6e", "#0284c7"),
    "emerald": ("#064e3b", "#059669"),
    "teal": ("#134e4a", "#0d9488"),
    "rose": ("#881337", "#e11d48"),
    "slate": ("#0f172a", "#475569"),
}
SIZES = {"og": (1200, 630), "hero": (1600, 900), "w800": (800, 450), "w400": (400, 225)}
XMP_AI = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    b'</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)
XMP_TEMPLATE = XMP_AI.replace(b"trainedAlgorithmicMedia", b"digitalCreation")


class ImageError(Exception):
    pass


@dataclass
class Generated:
    data: bytes
    media_type: str
    provider: str
    model: str
    prompt: str
    seed: int | None
    ai_generated: bool
    cost_usd: float = 0.0


@dataclass
class Brief:
    title: str
    category_label: str
    tone: str
    subject: str
    composition: str = ""
    style_notes: str = ""
    avoid: list[str] = field(default_factory=list)
    seed_key: str = ""


class Provider(Protocol):
    name: str

    async def generate(self, brief: Brief, prompt: str, negative: str, n: int) -> list[Generated]: ...


def build_prompt(brief: Brief, style_prompt: str, negative: str) -> tuple[str, str]:
    """No people, logos, trademarks or text: the negative prompt always carries these, whatever the profile says."""
    must_avoid = ["people", "faces", "hands", "logos", "trademarks", "brand names", "copyrighted characters", "text", "letters", "watermarks", "UI screenshots"]
    neg = ", ".join(dict.fromkeys([*must_avoid, *[a.strip() for a in brief.avoid if a.strip()], *[n.strip() for n in negative.split(",") if n.strip()]]))
    parts = [style_prompt.strip(), f"Subject: {brief.subject.strip() or brief.title}", brief.composition and f"Composition: {brief.composition}", brief.style_notes and f"Style: {brief.style_notes}", "Abstract technical illustration for a technology news article cover, wide 16:9, no text."]
    return " ".join(p for p in parts if p)[:1800], neg[:800]


# --- template provider ----------------------------------------------------------------------------------------------


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def render_template(brief: Brief, size: tuple[int, int] = (1600, 900), palette: dict[str, Any] | None = None) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    dark, light = (palette or {}).get(brief.tone) or TONES.get(brief.tone) or TONES["slate"]
    a, b = _hex(dark), _hex(light)
    img = Image.new("RGB", (w, h), a)
    px = ImageDraw.Draw(img)
    for x in range(0, w, 4):  # diagonal gradient, coarse columns are enough at these sizes
        t = x / w
        px.rectangle([x, 0, x + 4, h], fill=tuple(int(a[i] + (b[i] - a[i]) * t * 0.85) for i in range(3)))
    rng = random.Random(int(hashlib.sha256((brief.seed_key or brief.title).encode()).hexdigest()[:8], 16))  # noqa: S311 - art, not security
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Illustration: a node graph on the right (services, clusters, models: abstract, never a logo).
    nodes = [(int(w * (0.55 + rng.random() * 0.4)), int(h * (0.12 + rng.random() * 0.76))) for _ in range(14)]
    for i, p in enumerate(nodes):
        for q in nodes[i + 1 :]:
            if math.dist(p, q) < w * 0.17:
                od.line([p, q], fill=(255, 255, 255, 70), width=max(2, w // 600))
    for p in nodes:
        r = rng.randint(w // 160, w // 60)
        od.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 255, 255, 150), outline=(255, 255, 255, 220), width=2)
    for _ in range(3):
        cx, cy, r = int(w * rng.uniform(0.6, 1.0)), int(h * rng.uniform(0, 1)), int(w * rng.uniform(0.12, 0.3))
        od.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 40), width=max(2, w // 400))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    pad = int(w * 0.06)
    label = ImageFont.load_default(size=max(14, w // 55))
    d.rounded_rectangle([pad, pad, pad + int(d.textlength(brief.category_label.upper(), font=label)) + 32, pad + w // 55 + 22], radius=12, fill=(255, 255, 255))
    d.text((pad + 16, pad + 10), brief.category_label.upper(), font=label, fill=a)
    title_font = ImageFont.load_default(size=max(24, w // 22))
    lines = textwrap.wrap(brief.title, width=26, break_on_hyphens=False)[:4]
    y = int(h * 0.36)
    for line in lines:
        d.text((pad, y), line, font=title_font, fill=(255, 255, 255))
        y += int(w // 22 * 1.25)
    d.text((pad, h - pad - w // 60), "jithendar.com · News", font=ImageFont.load_default(size=max(12, w // 70)), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TemplateProvider:
    name = "template"

    def __init__(self, palette: dict[str, Any] | None = None):
        self.palette = palette

    async def generate(self, brief: Brief, prompt: str, negative: str, n: int) -> list[Generated]:
        return [Generated(render_template(brief, (1600, 900), self.palette), "image/png", "template", "template-v1", "", None, False, 0.0)]


class GeminiProvider:
    """Through the web app's existing Gemini integration (callback POST /news/images/generate)."""

    name = "gemini"

    def __init__(self, web: Any, price_usd: float):
        self.web = web
        self.price = price_usd

    async def generate(self, brief: Brief, prompt: str, negative: str, n: int) -> list[Generated]:
        out = []
        for i in range(n):
            r = await self.web.news_generate_image({"prompt": f"{prompt}\nAvoid: {negative}", "aspectRatio": "16:9", "variant": i + 1})
            if r.get("error"):
                raise ImageError(str(r["error"])[:200])
            out.append(Generated(base64.b64decode(r["data"]), r.get("mimeType", "image/png"), "gemini", str(r.get("model", "gemini")), prompt, None, True, self.price))
        return out


class OpenAIProvider:
    """OpenAI Images API (POST /v1/images/generations, b64_json output). Configured only with OPENAI_API_KEY."""

    name = "openai"
    URL = "https://api.openai.com/v1/images/generations"

    def __init__(self, api_key: str, model: str, price_usd: float, client: Any = None):
        if not api_key:
            raise ImageError("OPENAI_API_KEY is not set for the news agent")
        self.key, self.model, self.price, self.client = api_key, model, price_usd, client

    async def generate(self, brief: Brief, prompt: str, negative: str, n: int) -> list[Generated]:
        import httpx

        client = self.client or httpx.AsyncClient(timeout=120)
        try:
            r = await client.post(self.URL, headers={"Authorization": f"Bearer {self.key}"}, json={"model": self.model, "prompt": f"{prompt} Avoid: {negative}.", "n": n, "size": "1536x1024"})
        finally:
            if self.client is None:
                await client.aclose()
        if r.status_code >= 400:
            raise ImageError(f"OpenAI images {r.status_code}")
        data = r.json().get("data") or []
        return [Generated(base64.b64decode(d["b64_json"]), "image/png", "openai", self.model, prompt, None, True, self.price) for d in data if d.get("b64_json")]


# --- optimisation ---------------------------------------------------------------------------------------------------


def cover_variants(data: bytes, ai_generated: bool) -> dict[str, tuple[bytes, int, int]]:
    """Centre-crop to 1.905:1 (OG) and 16:9 (hero + responsive sizes), WebP, all metadata dropped except our XMP."""
    from PIL import Image, ImageOps

    try:
        src = Image.open(io.BytesIO(data))
        src.load()
    except Exception as e:
        raise ImageError(f"not a decodable image: {type(e).__name__}") from e
    if src.width < 400 or src.height < 225 or src.width * src.height > 40_000_000:
        raise ImageError(f"unusable dimensions {src.width}×{src.height}")
    base = ImageOps.exif_transpose(src).convert("RGB")  # pixels only: EXIF, ICC and text chunks are not carried over
    out: dict[str, tuple[bytes, int, int]] = {}
    for key, (w, h) in SIZES.items():
        im = ImageOps.fit(base, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=82, method=5, xmp=XMP_AI if ai_generated else XMP_TEMPLATE)
        out[key] = (buf.getvalue(), w, h)
    return out


def clamp_alt(text: str, lo: int = 80, hi: int = 125) -> tuple[str, bool]:
    t = " ".join(text.split()).strip().rstrip(".") + "."
    if len(t) > hi:
        cut = t[: hi - 1].rsplit(" ", 1)[0].rstrip(",;:")
        t = cut + "."
    return t, lo <= len(t) <= hi


VisionFn = Callable[[bytes, str], Awaitable[dict[str, Any]]]
