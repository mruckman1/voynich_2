"""Embed sign images using Gemini Embedding 2.

Gemini Embedding 2 is natively multimodal -- it maps images and text
into the same embedding space, enabling direct visual comparison.
"""

import os
import time

import numpy as np


def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key, value = key.strip(), value.strip()
                if not os.environ.get(key):
                    os.environ[key] = value


def _get_client():
    """Create Gemini client. Loads GEMINI_API_KEY from .env or environment."""
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env or export it.\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    from google import genai
    return genai.Client(api_key=api_key)


def embed_image(client, image_path, model="gemini-embedding-2-preview",
                output_dim=768):
    """Embed a single image. Returns L2-normalized embedding vector."""
    from google.genai import types

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
    mime_type = mime_map.get(ext, 'image/png')

    result = client.models.embed_content(
        model=model,
        contents=types.Content(
            parts=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        ),
        config=types.EmbedContentConfig(
            output_dimensionality=output_dim,
        ),
    )

    embedding = np.array(result.embeddings[0].values)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def embed_image_with_text(client, image_path, text_label,
                           model="gemini-embedding-2-preview",
                           output_dim=768):
    """Embed an image + text label together (interleaved multimodal)."""
    from google.genai import types

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
    mime_type = mime_map.get(ext, 'image/png')

    result = client.models.embed_content(
        model=model,
        contents=types.Content(
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part(text=f"Tachygraphic sign for syllable: {text_label}"),
            ]
        ),
        config=types.EmbedContentConfig(
            output_dimensionality=output_dim,
        ),
    )

    embedding = np.array(result.embeddings[0].values)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def _call_with_retry(fn, max_retries=5, base_delay=2.0):
    """Call fn() with exponential backoff on rate limit (429) errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            is_rate_limit = '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str
            if is_rate_limit and attempt < max_retries:
                wait = base_delay * (2 ** attempt)
                print(f"    Rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def embed_batch(client, items, mode='image_only',
                model="gemini-embedding-2-preview", output_dim=768,
                delay=1.0, delay_every=5):
    """Embed a batch of items with rate limiting and retry.

    Args:
        client: Gemini client
        items: List of dicts with 'name' and 'image_path' keys,
               and optionally 'text_label' for image+text mode
        mode: 'image_only' or 'image_text'
        model: Gemini model ID
        output_dim: Embedding dimension
        delay: Seconds to pause for rate limiting
        delay_every: Apply delay every N items

    Returns:
        Tuple of (names list, embeddings np.array, failed list)
    """
    names = []
    embeddings = []
    failed = []

    for i, item in enumerate(items):
        if i > 0 and i % delay_every == 0:
            print(f"  Embedded {i}/{len(items)}...")
            time.sleep(delay)

        try:
            if mode == 'image_text' and 'text_label' in item:
                emb = _call_with_retry(lambda: embed_image_with_text(
                    client, item['image_path'], item['text_label'],
                    model=model, output_dim=output_dim))
            else:
                emb = _call_with_retry(lambda: embed_image(
                    client, item['image_path'],
                    model=model, output_dim=output_dim))

            names.append(item['name'])
            embeddings.append(emb)
        except Exception as e:
            print(f"  Failed to embed {item['name']}: {e}")
            failed.append(item['name'])

    if embeddings:
        return names, np.array(embeddings), failed
    return names, np.zeros((0, output_dim)), failed
