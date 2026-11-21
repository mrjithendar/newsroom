# Cover images

`app/news/images.py`, `desk.make_cover`. Every news article gets a cover unless the job turns images off.

## Providers

| Provider | Default | Cost | Status |
|---|---|---|---|
| `template` | **yes** | free | Generated with Pillow: category colour gradient, an abstract node-graph illustration seeded by the article, the category label, the headline and "jithendar.com · News". Always available; the fallback for everything else. **Verified locally.** |
| `gemini` | no | per image (set `image_prices.gemini`; default estimate $0.04) | Through the web app's existing Gemini image integration (Admin → Integrations, `GEMINI_API_KEY` + `GEMINI_IMAGE_MODEL`). The agent never holds that key. **Mock-tested only** (needs the key). |
| `openai` | no | per image (`image_prices.openai`; default estimate $0.04) | OpenAI Images API `POST /v1/images/generations`, b64 output, 1536×1024, model `NEWS_OPENAI_IMAGE_MODEL` (default `gpt-image-1`; `gpt-image-2` exists but its sizes were not confirmed, N-15). Needs `OPENAI_API_KEY` in Secret `seo-agent-env`. **Mock-tested only.** |
| Local Stable Diffusion | not offered | — | On CPU-only Minikube one 1024² image takes minutes and several GiB of RAM; it would starve the cluster for a cover. |

Prices are configuration, not facts: check the provider's current price list and set Settings → image prices; the
budget uses them for its checks. The image budget (per job and global, per day and month) is checked before generating.

## Pipeline

1. **Brief**: the Art director writes subject/composition/style/avoid from the image profile and the article.
2. **Prompt**: profile style prompt + brief; the negative prompt **always** includes people, faces, hands, logos,
   trademarks, brand names, copyrighted characters, text, letters, watermarks and UI screenshots.
3. **Variants**: up to N (profile, default 2).
4. **Safety check**: a vision model (Haiku) looks at each variant: faces, logos/trademarks, text artifacts, other issues,
   match to the brief. Unsafe variants are rejected and recorded (`news_images.status = rejected_safety`).
5. **Best**: the safe variant with the best brief match; if none is safe or the provider fails → the template cover.
6. **Optimise**: centre-crop to 1200×630 (OG/social) and 16:9 (1600×900 hero, 800 and 400 wide), WebP q82. Only pixels
   are copied: EXIF, ICC and text chunks from the provider are dropped.
7. **Metadata**: XMP `Iptc4xmpExt:DigitalSourceType` = `trainedAlgorithmicMedia` for AI images, `digitalCreation` for
   template covers; `news_images.ai_generated`; the media item's description says "AI-generated image (provider)" or
   "Generated template cover (not AI)".
8. **Store**: the existing media library (R2, `storeUpload`). Without media storage the draft still goes to review,
   flagged "no cover".
9. **Alt text**: a vision model looks at the **final** 1200×630 image (not the prompt) and writes 80–125 characters
   describing what is visible; outside that range the reviewer is warned. "Decorative" verdicts are recorded (N-14).

In review: **Regenerate** (optional prompt tweak), **Use template cover**, pick an earlier stored cover, edit the alt text.
Each regeneration is a `news_image_flow` run with its own cost.

## Safety rules (never depict)

Real people, faces, logos, trademarks, brand marks, copyrighted characters, readable text. Source images are never
used. A cover is an abstract illustration of the topic.
