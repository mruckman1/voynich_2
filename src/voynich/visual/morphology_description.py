"""Method 1: Structured morphology description via LLM.

Use Gemini 3.1 Pro (via OpenRouter) to describe every sign in structured
terms, then compare descriptions as text.
"""

import asyncio
import base64
import json
import os
import time

import numpy as np

from voynich.visual.embed import _load_dotenv

MORPHOLOGY_SCHEMA = """\
Describe this handwritten tachygraphic sign using EXACTLY the following structured format.
Respond ONLY with the JSON object, no other text.

{
  "entry_direction": "up" | "down" | "left" | "right" | "curve_left" | "curve_right",
  "n_strokes": <integer 1-6>,
  "stroke_types": ["vertical" | "horizontal" | "diagonal_up" | "diagonal_down" |
                    "loop_clockwise" | "loop_counterclockwise" | "curve_left" |
                    "curve_right" | "hook_left" | "hook_right" | "dot"],
  "has_loop": true | false,
  "n_loops": <integer 0-3>,
  "loop_position": "top" | "middle" | "bottom" | "left" | "right" | null,
  "has_ascender": true | false,
  "has_descender": true | false,
  "terminal_shape": "hook_right" | "hook_left" | "open" | "closed" | "serif" |
                     "tapered" | "blunt" | "loop",
  "overall_height": "tall" | "medium" | "short",
  "overall_width": "wide" | "medium" | "narrow",
  "aspect_ratio": "portrait" | "square" | "landscape",
  "symmetry": "symmetric" | "left_heavy" | "right_heavy" | "asymmetric",
  "complexity": "simple" | "moderate" | "complex",
  "resembles_letter": "<closest Latin/Greek letter, or 'none'>",
  "distinctive_features": ["<free-text feature 1>", "<free-text feature 2>"]
}"""

MODEL_ID = "google/gemini-3.1-pro-preview"


def _get_openrouter_client():
    """Create OpenAI-compatible client pointed at OpenRouter."""
    _load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env or export it."
        )
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=4,
    )


async def describe_sign(client, image_path, sign_id):
    """Send a sign image to Gemini 3.1 Pro and get structured morphology."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode('utf-8')
    ext = image_path.rsplit('.', 1)[-1].lower()
    mime = 'image/png' if ext == 'png' else 'image/jpeg'

    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": MORPHOLOGY_SCHEMA},
            ],
        }],
    )

    text = response.choices[0].message.content
    text = text.replace('```json', '').replace('```', '').strip()
    morphology = json.loads(text)
    morphology['sign_id'] = sign_id
    return morphology


async def describe_all_signs(client, eva_items, costa_items,
                             max_concurrency=5):
    """Describe all EVA + Costamagna signs.

    Args:
        client: AsyncOpenAI client
        eva_items: list of dicts with 'image_path' and 'name'
        costa_items: list of dicts with 'image_path' and 'name'
        max_concurrency: max parallel API calls

    Returns: (eva_morphs, costa_morphs) — lists of morphology dicts
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def rate_limited(path, sign_id):
        async with semaphore:
            return await describe_sign(client, path, sign_id)

    eva_tasks = [
        rate_limited(item['image_path'], f"eva_{item['name']}")
        for item in eva_items
    ]
    costa_tasks = [
        rate_limited(item['image_path'], f"costa_{item['name']}")
        for item in costa_items
    ]

    print(f"  Describing {len(eva_tasks)} EVA + {len(costa_tasks)} Costamagna signs...")
    eva_results = await asyncio.gather(*eva_tasks, return_exceptions=True)
    costa_results = await asyncio.gather(*costa_tasks, return_exceptions=True)

    eva_morphs = [r for r in eva_results if isinstance(r, dict)]
    costa_morphs = [r for r in costa_results if isinstance(r, dict)]

    n_failed = (len(eva_results) - len(eva_morphs)
                + len(costa_results) - len(costa_morphs))
    if n_failed > 0:
        print(f"  {n_failed} descriptions failed")

    return eva_morphs, costa_morphs


def morphological_distance(morph_a, morph_b):
    """Compute distance between two morphology descriptions.

    Weighted sum of feature-level disagreements:
    - Categorical: 0 if same, 1 if different
    - Numeric: |a - b| / max_range
    - Set: 1 - Jaccard similarity
    - Boolean: 0 if same, 1 if different
    """
    weights = {
        'entry_direction': 2.0,
        'n_strokes': 1.5,
        'has_loop': 2.0,
        'n_loops': 1.5,
        'loop_position': 1.0,
        'has_ascender': 1.5,
        'has_descender': 1.5,
        'terminal_shape': 2.0,
        'overall_height': 0.5,
        'overall_width': 0.5,
        'aspect_ratio': 0.5,
        'symmetry': 1.0,
        'complexity': 0.5,
        'stroke_types': 1.5,
    }

    total_dist = 0.0
    total_weight = 0.0

    for feature, weight in weights.items():
        a_val = morph_a.get(feature)
        b_val = morph_b.get(feature)

        if a_val is None or b_val is None:
            continue

        if feature == 'stroke_types':
            set_a = set(a_val) if isinstance(a_val, list) else set()
            set_b = set(b_val) if isinstance(b_val, list) else set()
            union = set_a | set_b
            dist = 1 - len(set_a & set_b) / len(union) if union else 0
        elif feature in ('n_strokes', 'n_loops'):
            max_val = 6 if feature == 'n_strokes' else 3
            dist = abs(a_val - b_val) / max_val
        elif isinstance(a_val, bool):
            dist = 0.0 if a_val == b_val else 1.0
        else:
            dist = 0.0 if a_val == b_val else 1.0

        total_dist += weight * dist
        total_weight += weight

    return total_dist / total_weight if total_weight > 0 else 1.0


def build_morphology_matrix(eva_morphs, costa_morphs):
    """Build distance matrix from morphological descriptions.

    Returns (n_eva x n_costa) distance matrix. Lower = more similar.
    """
    n_eva = len(eva_morphs)
    n_costa = len(costa_morphs)
    matrix = np.zeros((n_eva, n_costa))

    for i, eva_m in enumerate(eva_morphs):
        for j, costa_m in enumerate(costa_morphs):
            matrix[i, j] = morphological_distance(eva_m, costa_m)

    return matrix
