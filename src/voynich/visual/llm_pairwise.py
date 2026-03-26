"""Method 7: Direct LLM pairwise comparison.

Show Gemini 3.1 Pro two signs side by side and ask for a structured
comparison. 25 T_P15 proposed + 100 random controls = 125 calls.
"""

import asyncio
import base64
import json
import os
import random

import numpy as np

from voynich.visual.embed import _load_dotenv

PAIRWISE_PROMPT = """\
You are a paleography expert comparing two handwritten shorthand signs.

Sign A (left) is from a 15th century manuscript.
Sign B (right) is from a documented medieval Italian tachygraphic syllabary.

Compare these two signs on the following dimensions. Respond ONLY with the JSON object.

{
  "same_basic_structure": true | false,
  "shared_features": ["<feature 1>", "<feature 2>"],
  "different_features": ["<feature 1>", "<feature 2>"],
  "entry_angle_match": true | false,
  "stroke_count_match": true | false,
  "loop_match": true | false,
  "terminal_match": true | false,
  "ascender_descender_match": true | false,
  "overall_similarity": 0.0 to 1.0,
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one sentence explaining the comparison>"
}"""

MODEL_ID = "google/gemini-3.1-pro-preview"


def _get_openrouter_client():
    """Create AsyncOpenAI client for OpenRouter."""
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


async def compare_pair(client, eva_path, costa_path, eva_name,
                       costa_syllable):
    """Show two sign images to Gemini 3.1 Pro for comparison."""
    with open(eva_path, 'rb') as f:
        eva_b64 = base64.b64encode(f.read()).decode('utf-8')
    with open(costa_path, 'rb') as f:
        costa_b64 = base64.b64encode(f.read()).decode('utf-8')

    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{eva_b64}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{costa_b64}"}},
                {"type": "text", "text": PAIRWISE_PROMPT},
            ],
        }],
    )

    text = response.choices[0].message.content
    text = text.replace('```json', '').replace('```', '').strip()
    result = json.loads(text)
    result['eva_name'] = eva_name
    result['costamagna_syllable'] = costa_syllable
    return result


async def run_t_p15_pairwise(client, eva_items, costa_items, t_p15_table,
                             n_controls=4, max_concurrency=5, seed=42):
    """For each T_P15 assignment, compare proposed + random controls.

    Args:
        client: AsyncOpenAI client
        eva_items: list of dicts with 'name', 'image_path'
        costa_items: list of dicts with 'name', 'image_path'
        t_p15_table: dict mapping EVA name -> proposed syllable
        n_controls: number of random control comparisons per EVA char
        max_concurrency: max parallel API calls
        seed: random seed for control selection

    Returns: list of comparison dicts, each with 'comparison_type' field
    """
    rng = random.Random(seed)
    semaphore = asyncio.Semaphore(max_concurrency)

    eva_by_name = {item['name']: item for item in eva_items}
    costa_by_name = {item['name']: item for item in costa_items}

    async def rate_limited(eva_p, costa_p, eva_n, costa_s):
        async with semaphore:
            return await compare_pair(client, eva_p, costa_p, eva_n, costa_s)

    tasks = []
    task_types = []

    for eva_name, proposed in t_p15_table.items():
        if eva_name not in eva_by_name:
            continue
        eva_img = eva_by_name[eva_name]['image_path']

        # Find Costamagna image for proposed syllable
        costa_match = None
        for cname, citem in costa_by_name.items():
            if cname == proposed or proposed in cname.split('-'):
                costa_match = citem
                break

        if not costa_match:
            continue

        # Proposed comparison
        tasks.append(rate_limited(eva_img, costa_match['image_path'],
                                  eva_name, proposed))
        task_types.append('T_P15_PROPOSED')

        # Random controls
        available = [c for c in costa_items
                     if c['name'] != proposed
                     and proposed not in c['name'].split('-')]
        controls = rng.sample(available, min(n_controls, len(available)))
        for ctrl in controls:
            tasks.append(rate_limited(eva_img, ctrl['image_path'],
                                      eva_name, ctrl['name']))
            task_types.append('RANDOM_CONTROL')

    print(f"  Running {len(tasks)} pairwise comparisons...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    comparisons = []
    for result, ctype in zip(results, task_types):
        if isinstance(result, dict):
            result['comparison_type'] = ctype
            comparisons.append(result)

    n_failed = sum(1 for r in results if not isinstance(r, dict))
    if n_failed > 0:
        print(f"  {n_failed} comparisons failed")

    return comparisons


def score_pairwise_results(comparisons):
    """Score pairwise results: does the LLM rate proposed > controls?

    Returns dict with per-EVA results and aggregate statistics.
    """
    by_eva = {}
    for comp in comparisons:
        eva = comp['eva_name']
        if eva not in by_eva:
            by_eva[eva] = {'proposed': None, 'controls': []}
        if comp['comparison_type'] == 'T_P15_PROPOSED':
            by_eva[eva]['proposed'] = comp
        else:
            by_eva[eva]['controls'].append(comp)

    results = {}
    for eva, data in by_eva.items():
        if not data['proposed'] or not data['controls']:
            continue

        proposed_sim = data['proposed'].get('overall_similarity', 0)
        control_sims = [c.get('overall_similarity', 0)
                        for c in data['controls']]
        mean_control = float(np.mean(control_sims)) if control_sims else 0

        rank = sum(1 for c in control_sims if c >= proposed_sim) + 1

        results[eva] = {
            'proposed_similarity': proposed_sim,
            'mean_control_similarity': mean_control,
            'rank_among_5': rank,
            'proposed_beats_controls': proposed_sim > mean_control,
            'same_basic_structure': data['proposed'].get(
                'same_basic_structure', False),
            'reasoning': data['proposed'].get('reasoning', ''),
            'shared_features': data['proposed'].get('shared_features', []),
        }

    n_wins = sum(1 for r in results.values() if r['proposed_beats_controls'])
    n_structure = sum(1 for r in results.values()
                      if r['same_basic_structure'])

    return {
        'per_eva': results,
        'n_tested': len(results),
        'n_proposed_wins': n_wins,
        'win_rate': n_wins / len(results) if results else 0,
        'n_same_structure': n_structure,
        'mean_proposed_sim': float(np.mean(
            [r['proposed_similarity'] for r in results.values()]
        )) if results else 0,
        'mean_control_sim': float(np.mean(
            [r['mean_control_similarity'] for r in results.values()]
        )) if results else 0,
    }
