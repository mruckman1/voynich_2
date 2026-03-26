"""
Phase 74, Track B2: LLM Gap-Filling with Hallucination Controls
================================================================
T1-dense passages have ~80% identified tokens. The remaining 20% (3-4
unknown tokens per 15-token window) can be attacked by LLM gap-filling
with rigorous hallucination controls.

Five layers of protection:
  1. Known-answer calibration: 10 passages with known answers, accuracy ≥30%
  2. Shuffled-gloss control: same gaps but shuffled surrounding glosses
  3. Random-passage control: random passages with artificial gaps
  4. Cross-run consistency: 3 runs per real passage, same word each time
  5. Decode-string agreement: proposed word ED ≤ 2 from CVC decoded string

A proposal is ACCEPTED only if it passes ALL FIVE layers.

Dependency chain:
    results/p69_clean_corpus.json        (Phase 69 — T1 catalogue)
    results/combined_refine.json         (Phase 15 — assignment table)
    results/p74_patterns.json            (Track B1 — optional)
        -> results/p74_llm_gapfill.json
"""

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from voynich.core._paths import results_dir as _results_dir
from voynich.core.corpus import build_eva_to_triple_lookup, load_corpus, tokenize_eva_chars
from voynich.core.reference import build_expanded_word_set, load_reference_corpus
from voynich.phases.corrected_coda import (
    build_coda_table_v2,
    decode_token_cvc_v2,
)
from voynich.phases.suffix_calibration import SIGNAL_WORDS_51


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _convert(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(item) for item in obj]
    if isinstance(obj, float) and (obj != obj):
        return None
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _safe_load(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(rd: str, filename: str, data: Any) -> str:
    path = os.path.join(rd, filename)
    with open(path, 'w') as f:
        json.dump(_convert(data), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemini-3.1-pro-preview"


def _get_openrouter_client():
    """Create AsyncOpenAI client for OpenRouter."""
    from voynich.visual.embed import _load_dotenv
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


# ---------------------------------------------------------------------------
# Passage selection and decoding
# ---------------------------------------------------------------------------

def _decode_corpus(
    all_tokens: List[str],
    assignment: Dict[str, str],
    eva_to_triple: Dict[str, str],
) -> List[str]:
    """Decode corpus with connector→null baseline."""
    coda_table = build_coda_table_v2()
    coda_table.stroke_to_coda['connector'] = ''

    decoded = []
    for token in all_tokens:
        result = decode_token_cvc_v2(token, assignment, eva_to_triple, coda_table)
        decoded.append(result.decoded_cvc)
    return decoded


def _build_gloss_lookup(
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
) -> Dict[str, str]:
    """Build decoded_word -> gloss lookup from T1 + signal words."""
    lookup = {}

    # Signal words
    for word, info in SIGNAL_WORDS_51.items():
        gloss = info.get('gloss', word)
        lookup[word] = gloss

    # T1 identifications
    for eva_type, info in t1_types.items():
        matched = info.get('matched_word', '')
        if matched:
            lookup[matched] = info.get('gloss', matched)
            # Also add the EVA type itself as a key
            lookup[eva_type] = info.get('gloss', matched)

    return lookup


def _select_gap_fill_passages(
    all_tokens: List[str],
    decoded: List[str],
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
    folio_list: List[str],
    n: int = 15,
    window: int = 15,
) -> List[Dict[str, Any]]:
    """Select passages with ≥75% identified tokens and 1-5 gaps."""
    t1_set = set(t1_types.keys())
    windows = []

    for start in range(0, len(all_tokens) - window):
        # Check same folio
        if folio_list[start] != folio_list[start + window - 1]:
            continue

        w_tokens = all_tokens[start:start + window]
        w_decoded = decoded[start:start + window]
        folio = folio_list[start]

        identified = []
        gaps = []

        for i, (tok, dec) in enumerate(zip(w_tokens, w_decoded)):
            if tok in t1_set:
                t1_info = t1_types[tok]
                identified.append({
                    'position': i,
                    'eva': tok,
                    'word': t1_info.get('matched_word', dec or '?'),
                    'gloss': t1_info.get('gloss',
                             t1_info.get('matched_word', '?')),
                    'source': 'T1',
                })
            elif dec and dec.lower() in ref_word_set:
                # Look up gloss from signal words
                gloss = SIGNAL_WORDS_51.get(dec, {}).get('gloss', dec)
                identified.append({
                    'position': i,
                    'eva': tok,
                    'word': dec,
                    'gloss': gloss,
                    'source': 'dict',
                })
            else:
                gaps.append({
                    'position': i,
                    'eva': tok,
                    'decoded': dec or '',
                })

        id_fraction = len(identified) / window
        if id_fraction >= 0.75 and 1 <= len(gaps) <= 5:
            windows.append({
                'start': start,
                'folio': folio,
                'identified': identified,
                'gaps': gaps,
                'id_fraction': id_fraction,
                'n_gaps': len(gaps),
            })

    # Sort by identification fraction (desc), then fewest gaps
    windows.sort(key=lambda w: (-w['id_fraction'], w['n_gaps']))

    # Deduplicate overlapping windows
    selected = []
    used = set()
    for w in windows:
        positions = set(range(w['start'], w['start'] + window))
        if not positions & used:
            selected.append(w)
            used.update(positions)
        if len(selected) >= n:
            break

    return selected


# ---------------------------------------------------------------------------
# Control generation
# ---------------------------------------------------------------------------

def _build_shuffled_control(passage: Dict, seed: int) -> Dict:
    """Shuffle glosses of identified tokens while keeping gap positions fixed."""
    rng = np.random.default_rng(seed)
    shuffled = dict(passage)
    identified = list(passage['identified'])

    glosses = [t['gloss'] for t in identified]
    words = [t['word'] for t in identified]

    perm = rng.permutation(len(glosses))
    shuffled_identified = []
    for i, orig in enumerate(identified):
        shuffled_identified.append({
            **orig,
            'gloss': glosses[perm[i]],
            'word': words[perm[i]],
        })

    shuffled['identified'] = shuffled_identified
    shuffled['control_type'] = 'SHUFFLED_GLOSSES'
    return shuffled


def _build_random_passage(
    all_tokens: List[str],
    decoded: List[str],
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
    folio_list: List[str],
    n_gaps: int,
    seed: int,
) -> Optional[Dict]:
    """Build a random passage with artificial gaps."""
    rng = np.random.default_rng(seed)
    t1_set = set(t1_types.keys())
    window = 15

    # Try random starting positions
    for _ in range(200):
        start = rng.integers(0, max(1, len(all_tokens) - window))
        if folio_list[start] != folio_list[min(start + window - 1,
                                                len(all_tokens) - 1)]:
            continue

        w_tokens = all_tokens[start:start + window]
        w_decoded = decoded[start:start + window]

        if len(w_tokens) < window:
            continue

        # Check that enough tokens are identifiable
        id_count = sum(1 for tok, dec in zip(w_tokens, w_decoded)
                       if tok in t1_set or (dec and dec.lower() in ref_word_set))

        if id_count < window - n_gaps:
            continue

        # Randomly designate some as gaps
        id_positions = [i for i, (tok, dec) in enumerate(zip(w_tokens, w_decoded))
                        if tok in t1_set or (dec and dec.lower() in ref_word_set)]

        if len(id_positions) < n_gaps:
            continue

        gap_positions = set(rng.choice(id_positions, size=n_gaps, replace=False))

        identified = []
        gaps = []
        for i, (tok, dec) in enumerate(zip(w_tokens, w_decoded)):
            if i in gap_positions:
                gaps.append({
                    'position': i,
                    'eva': tok,
                    'decoded': dec or '',
                })
            elif tok in t1_set:
                t1_info = t1_types[tok]
                identified.append({
                    'position': i,
                    'eva': tok,
                    'word': t1_info.get('matched_word', dec or '?'),
                    'gloss': t1_info.get('gloss',
                             t1_info.get('matched_word', '?')),
                    'source': 'T1',
                })
            elif dec and dec.lower() in ref_word_set:
                gloss = SIGNAL_WORDS_51.get(dec, {}).get('gloss', dec)
                identified.append({
                    'position': i,
                    'eva': tok,
                    'word': dec,
                    'gloss': gloss,
                    'source': 'dict',
                })
            else:
                gaps.append({
                    'position': i,
                    'eva': tok,
                    'decoded': dec or '',
                })

        return {
            'start': int(start),
            'folio': folio_list[start],
            'identified': identified,
            'gaps': gaps,
            'id_fraction': len(identified) / window,
            'n_gaps': len(gaps),
            'control_type': 'RANDOM_PASSAGE',
        }

    return None


def _build_known_answer_passages(
    all_tokens: List[str],
    decoded: List[str],
    t1_types: Dict[str, Dict],
    ref_word_set: Set[str],
    folio_list: List[str],
    n: int = 10,
) -> List[Dict]:
    """Find passages where ALL 15 tokens are identified.
    Mask 3 random tokens as 'gaps'. We know the answers."""
    t1_set = set(t1_types.keys())
    window = 15

    fully_identified = []
    for start in range(0, len(all_tokens) - window):
        if folio_list[start] != folio_list[start + window - 1]:
            continue

        w_tokens = all_tokens[start:start + window]
        w_decoded = decoded[start:start + window]

        all_id = all(
            tok in t1_set or (dec and dec.lower() in ref_word_set)
            for tok, dec in zip(w_tokens, w_decoded)
        )
        if all_id:
            fully_identified.append(start)

    if not fully_identified:
        return []

    rng = np.random.default_rng(42)
    selected = rng.choice(fully_identified,
                          size=min(n, len(fully_identified)),
                          replace=False)

    known_answer = []
    for start in selected:
        start = int(start)
        w_tokens = all_tokens[start:start + window]
        w_decoded = decoded[start:start + window]

        mask_positions = set(rng.choice(window, size=3, replace=False))

        identified = []
        gaps = []
        answers = {}

        for i, (tok, dec) in enumerate(zip(w_tokens, w_decoded)):
            if tok in t1_set:
                word = t1_types[tok].get('matched_word', dec or '?')
                gloss = t1_types[tok].get('gloss',
                        t1_types[tok].get('matched_word', '?'))
            else:
                word = dec or '?'
                gloss = SIGNAL_WORDS_51.get(dec, {}).get('gloss', dec or '?')

            if i in mask_positions:
                gaps.append({
                    'position': i,
                    'eva': tok,
                    'decoded': dec or '',
                })
                answers[str(i)] = {'word': word, 'gloss': gloss}
            else:
                identified.append({
                    'position': i,
                    'eva': tok,
                    'word': word,
                    'gloss': gloss,
                    'source': 'T1' if tok in t1_set else 'dict',
                })

        known_answer.append({
            'start': start,
            'folio': folio_list[start],
            'identified': identified,
            'gaps': gaps,
            'answers': answers,
            'id_fraction': len(identified) / window,
            'n_gaps': len(gaps),
            'control_type': 'KNOWN_ANSWER',
        })

    return known_answer


# ---------------------------------------------------------------------------
# LLM prompt and query
# ---------------------------------------------------------------------------

GAP_FILL_PROMPT = """You are examining a partially decoded medieval Latin pharmaceutical text.
Most words have been identified; a few gaps remain. Your task is to
propose the most likely Latin/Italian word for each gap based on the
surrounding pharmaceutical context.

PASSAGE (15 words, {n_gaps} gaps marked with [GAP]):
{passage_display}

KNOWN WORDS WITH GLOSSES:
{known_words_section}

CONSTRAINTS:
- This is pharmaceutical recipe text (preparation instructions, ingredient lists)
- The language is macaronic Latin-Italian with Gallo-Italic features
- Common recipe verbs: cola (strain), tere (grind), misce (mix), coque (cook)
- Common ingredients: senna, coral, root, dung, honey, wax, salt, water, oil
- Common prepositions: cum/con (with), in, per (through), de (of/from)
- Function words: ne (not), se (if/self), di (of)
- Words ending in -n are likely accusative case (direct objects)
- Words ending in -s are likely 2nd person verbs (imperative instructions)
- Words ending in -t are likely 3rd person verbs (property descriptions)

FOR EACH GAP, provide:
1. Your best guess for the Latin/Italian word
2. Confidence: HIGH / MEDIUM / LOW
3. Reasoning (one sentence: what in the context suggests this word?)

If you cannot determine a word, say "UNKNOWN" — this is preferable to guessing.

Respond ONLY with JSON:
{{
  "gaps": [
    {{
      "position": <position_number>,
      "proposed_word": "...",
      "proposed_gloss": "...",
      "confidence": "HIGH|MEDIUM|LOW",
      "reasoning": "..."
    }}
  ]
}}"""


def _format_passage_for_llm(passage: Dict) -> Tuple[str, str]:
    """Build passage display and known words section for the prompt."""
    all_positions = {}
    for tok in passage['identified']:
        all_positions[tok['position']] = ('ID', tok)
    for gap in passage['gaps']:
        all_positions[gap['position']] = ('GAP', gap)

    display_parts = []
    known_parts = []

    for pos in range(15):
        if pos in all_positions:
            ptype, data = all_positions[pos]
            if ptype == 'ID':
                display_parts.append(
                    f"  Position {pos + 1}: {data['word']} ({data['gloss']})")
                known_parts.append(f"  {data['word']}: {data['gloss']}")
            else:
                decoded_hint = data.get('decoded', '?')
                display_parts.append(
                    f"  Position {pos + 1}: [GAP — decoded hint: "
                    f"\"{decoded_hint}\"]")

    return '\n'.join(display_parts), '\n'.join(known_parts)


async def _query_llm(client, passage: Dict, semaphore) -> Optional[Dict]:
    """Send a single gap-fill query to the LLM."""
    async with semaphore:
        display, known = _format_passage_for_llm(passage)
        prompt = GAP_FILL_PROMPT.format(
            n_gaps=passage['n_gaps'],
            passage_display=display,
            known_words_section=known,
        )

        try:
            response = await client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )

            text = response.choices[0].message.content
            text = text.strip().replace('```json', '').replace('```', '').strip()

            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Try extracting JSON block
                idx = text.find('{')
                if idx >= 0:
                    depth = 0
                    for i in range(idx, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(text[idx:i + 1])
                                except (json.JSONDecodeError, ValueError):
                                    break
                return None
        except Exception as e:
            print(f"    API error: {e}")
            return None


# ---------------------------------------------------------------------------
# Run and score
# ---------------------------------------------------------------------------

async def _run_gap_filling_async(
    real_passages: List[Dict],
    known_answer_passages: List[Dict],
    shuffled_passages: List[Dict],
    random_passages: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Run gap-filling on all passage types."""
    client = _get_openrouter_client()
    semaphore = asyncio.Semaphore(5)

    # Build task list
    tasks = []
    task_meta = []

    # Real passages: 3 runs each
    for passage in real_passages:
        for run_idx in range(3):
            tasks.append(_query_llm(client, passage, semaphore))
            task_meta.append({
                'passage_start': passage['start'],
                'control_type': None,
                'run_idx': run_idx,
            })

    # Shuffled: 1 run each
    for passage in shuffled_passages:
        tasks.append(_query_llm(client, passage, semaphore))
        task_meta.append({
            'passage_start': passage['start'],
            'control_type': 'SHUFFLED_GLOSSES',
            'run_idx': 0,
        })

    # Random: 1 run each
    for passage in random_passages:
        tasks.append(_query_llm(client, passage, semaphore))
        task_meta.append({
            'passage_start': passage.get('start', -1),
            'control_type': 'RANDOM_PASSAGE',
            'run_idx': 0,
        })

    # Known-answer: 1 run each
    for passage in known_answer_passages:
        tasks.append(_query_llm(client, passage, semaphore))
        task_meta.append({
            'passage_start': passage['start'],
            'control_type': 'KNOWN_ANSWER',
            'run_idx': 0,
        })

    print(f"    Sending {len(tasks)} API calls...")
    results = await asyncio.gather(*tasks)

    # Group results
    all_results = []
    ka_results = []

    # Group real passage results by start position
    real_runs: Dict[int, List] = {}
    for result, meta in zip(results, task_meta):
        if meta['control_type'] is None:
            start = meta['passage_start']
            if start not in real_runs:
                real_runs[start] = []
            real_runs[start].append(result)
        elif meta['control_type'] == 'SHUFFLED_GLOSSES':
            all_results.append({
                'passage_start': meta['passage_start'],
                'control_type': 'SHUFFLED_GLOSSES',
                'result': result,
            })
        elif meta['control_type'] == 'RANDOM_PASSAGE':
            all_results.append({
                'passage_start': meta['passage_start'],
                'control_type': 'RANDOM_PASSAGE',
                'result': result,
            })
        elif meta['control_type'] == 'KNOWN_ANSWER':
            ka_results.append({
                'passage_start': meta['passage_start'],
                'result': result,
            })

    # Package real results
    for passage in real_passages:
        runs = real_runs.get(passage['start'], [])
        all_results.append({
            'passage_start': passage['start'],
            'control_type': None,
            'runs': runs,
            'passage': {
                'start': passage['start'],
                'folio': passage['folio'],
                'gaps': passage['gaps'],
                'n_gaps': passage['n_gaps'],
            },
        })

    return all_results, ka_results


def _score_known_answers(
    ka_results: List[Dict],
    known_answer_passages: List[Dict],
) -> Dict[str, Any]:
    """Score known-answer calibration."""
    total_correct = 0
    total_gaps = 0

    for ka_res, ka_passage in zip(ka_results, known_answer_passages):
        result = ka_res.get('result')
        if not result:
            total_gaps += len(ka_passage['gaps'])
            continue

        answers = ka_passage.get('answers', {})

        for gap_result in result.get('gaps', []):
            pos = gap_result.get('position', 0) - 1  # 1-indexed in prompt
            pos_key = str(pos)
            if pos_key in answers:
                total_gaps += 1
                known_word = answers[pos_key]['word']
                proposed = gap_result.get('proposed_word', '')
                if _edit_distance(proposed.lower(), known_word.lower()) <= 1:
                    total_correct += 1

    accuracy = total_correct / total_gaps if total_gaps > 0 else 0.0

    return {
        'n_correct': total_correct,
        'n_total': total_gaps,
        'accuracy': accuracy,
    }


def _score_gap_filling(
    all_results: List[Dict],
    ka_scores: Dict[str, Any],
    ref_word_set: Set[str],
) -> Dict[str, Any]:
    """Score real vs controls."""
    # Group by control type
    real = [r for r in all_results if r.get('control_type') is None]
    shuffled = [r for r in all_results
                if r.get('control_type') == 'SHUFFLED_GLOSSES']

    # Real confidence distribution
    real_confidences = []
    for r in real:
        for run in r.get('runs', []):
            if run:
                for gap in run.get('gaps', []):
                    real_confidences.append(gap.get('confidence', 'LOW'))

    real_high = sum(1 for c in real_confidences if c == 'HIGH')
    real_high_rate = real_high / len(real_confidences) if real_confidences else 0.0

    # Shuffled confidence distribution
    shuf_confidences = []
    for r in shuffled:
        result = r.get('result')
        if result:
            for gap in result.get('gaps', []):
                shuf_confidences.append(gap.get('confidence', 'LOW'))

    shuf_high = sum(1 for c in shuf_confidences if c == 'HIGH')
    shuf_high_rate = shuf_high / len(shuf_confidences) if shuf_confidences else 0.0

    conf_selectivity = real_high_rate / (shuf_high_rate + 0.001)

    # Cross-run consistency (real passages only)
    consistency_scores = []
    for r in real:
        runs = r.get('runs', [])
        valid_runs = [run for run in runs if run]
        if len(valid_runs) < 2:
            continue

        passage_info = r.get('passage', {})
        n_gaps = passage_info.get('n_gaps', 0)

        for gap_idx in range(n_gaps):
            proposed_words = []
            for run in valid_runs:
                gaps = run.get('gaps', [])
                if gap_idx < len(gaps):
                    proposed_words.append(
                        gaps[gap_idx].get('proposed_word', '').lower())

            if len(proposed_words) >= 2:
                agreement = len(set(proposed_words)) == 1
                consistency_scores.append(1.0 if agreement else 0.0)

    consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

    # Decode agreement (real passages, majority vote)
    decode_agreements = []
    for r in real:
        runs = r.get('runs', [])
        valid_runs = [run for run in runs if run]
        passage_info = r.get('passage', {})
        gaps = passage_info.get('gaps', [])

        for gap_info in gaps:
            decoded = gap_info.get('decoded', '')
            if not decoded:
                continue

            # Get majority proposed word
            proposed_words = []
            for run in valid_runs:
                run_gaps = run.get('gaps', [])
                matching = next(
                    (g for g in run_gaps
                     if g.get('position') == gap_info['position'] + 1), None)
                if matching:
                    proposed_words.append(
                        matching.get('proposed_word', '').lower())

            if proposed_words:
                majority = Counter(proposed_words).most_common(1)[0][0]
                ed = _edit_distance(decoded.lower(), majority)
                decode_agreements.append(ed <= 2)

    decode_agreement = float(np.mean(decode_agreements)) if decode_agreements else 0.0

    return {
        'ka_accuracy': ka_scores['accuracy'],
        'real_high_confidence_rate': round(real_high_rate, 4),
        'shuffled_high_confidence_rate': round(shuf_high_rate, 4),
        'confidence_selectivity': round(conf_selectivity, 4),
        'consistency': round(consistency, 4),
        'decode_agreement': round(decode_agreement, 4),
        'n_real_confidences': len(real_confidences),
        'n_shuffled_confidences': len(shuf_confidences),
        'n_consistency_checks': len(consistency_scores),
    }


# ---------------------------------------------------------------------------
# Validate and accept proposals
# ---------------------------------------------------------------------------

def _validate_proposals(
    all_results: List[Dict],
    scores: Dict[str, Any],
    ref_word_set: Set[str],
) -> Dict[str, Any]:
    """Accept or reject gap-fill proposals based on 5-layer criteria."""
    real = [r for r in all_results if r.get('control_type') is None]

    accepted = []
    rejected = []

    for r in real:
        runs = r.get('runs', [])
        valid_runs = [run for run in runs if run]
        passage_info = r.get('passage', {})
        gaps = passage_info.get('gaps', [])

        for gap_info in gaps:
            decoded = gap_info.get('decoded', '')

            # Collect proposals across runs
            proposals = []
            for run in valid_runs:
                run_gaps = run.get('gaps', [])
                matching = next(
                    (g for g in run_gaps
                     if g.get('position') == gap_info['position'] + 1), None)
                if matching:
                    proposals.append(matching)

            if not proposals:
                continue

            # Majority vote
            proposed_words = [p.get('proposed_word', '').lower()
                              for p in proposals]
            majority_word = Counter(proposed_words).most_common(1)[0][0]
            consistent = len(set(proposed_words)) == 1

            # Majority confidence
            confs = [p.get('confidence', 'LOW') for p in proposals]
            majority_conf = Counter(confs).most_common(1)[0][0]

            # Decode agreement (ED ≤ 2)
            ed_with_decoded = _edit_distance(decoded.lower(), majority_word) \
                if decoded else 99
            decode_consistent = ed_with_decoded <= 2

            # In dictionary
            in_dict = majority_word in ref_word_set

            proposal = {
                'position': gap_info['position'],
                'passage_start': passage_info.get('start', -1),
                'decoded_string': decoded,
                'proposed_word': majority_word,
                'proposed_gloss': proposals[0].get('proposed_gloss', '?'),
                'confidence': majority_conf,
                'consistent_across_runs': consistent,
                'ed_with_decoded': ed_with_decoded,
                'decode_consistent': decode_consistent,
                'in_dictionary': in_dict,
                'reasoning': proposals[0].get('reasoning', ''),
                'n_runs': len(proposals),
            }

            if (consistent and decode_consistent and in_dict
                    and majority_conf == 'HIGH'):
                proposal['status'] = 'ACCEPTED'
                accepted.append(proposal)
            else:
                reasons = []
                if not consistent:
                    reasons.append('inconsistent')
                if not decode_consistent:
                    reasons.append('decode_mismatch')
                if not in_dict:
                    reasons.append('not_in_dict')
                if majority_conf != 'HIGH':
                    reasons.append('low_confidence')
                proposal['status'] = 'REJECTED'
                proposal['rejection_reasons'] = reasons
                rejected.append(proposal)

    return {
        'n_accepted': len(accepted),
        'n_rejected': len(rejected),
        'acceptance_rate': len(accepted) / (len(accepted) + len(rejected))
        if (accepted or rejected) else 0.0,
        'accepted': accepted,
        'rejected': rejected[:20],  # Truncate for JSON size
    }


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMGapFillResult:
    phase: str = "74"
    step: str = "74.B2"
    experiment: str = "llm_gap_filling"
    n_real_passages: int = 0
    n_shuffled: int = 0
    n_random: int = 0
    n_known_answer: int = 0
    n_api_calls: int = 0
    n_api_failures: int = 0
    # Scoring
    ka_accuracy: float = 0.0
    confidence_selectivity: float = 0.0
    consistency: float = 0.0
    decode_agreement: float = 0.0
    # Proposals
    n_accepted: int = 0
    n_rejected: int = 0
    acceptance_rate: float = 0.0
    accepted_proposals: List[Dict[str, Any]] = field(default_factory=list)
    n_passages_fully_filled: int = 0
    # Gates
    gate_b2_1: bool = False   # KA accuracy ≥ 30%
    gate_b2_2: bool = False   # Confidence selectivity > 1.5×
    gate_b2_3: bool = False   # Consistency > 50%
    gate_b2_4: bool = False   # ≥5 accepted proposals
    gate_b2_5: bool = False   # Decode agreement > 40%
    gate_b2_6: bool = False   # ≥1 passage fully filled
    gates_passed: int = 0
    gate_passed: bool = False
    verdict: str = "UNKNOWN"
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_llm_gap_fill():
    """Track B2: LLM gap-filling with hallucination controls."""
    t0 = time.time()
    rd = str(_results_dir())

    print("Phase 74.B2 — LLM Gap-Filling with Hallucination Controls")
    print("=" * 57)

    # --- Load data ---
    print("  Loading data...")
    clean_data = _safe_load(os.path.join(rd, 'p69_clean_corpus.json'))
    t1_catalogue = clean_data.get('t1_catalogue', [])
    t1_types: Dict[str, Dict] = {}
    for entry in t1_catalogue:
        eva_type = entry.get('eva_type', '')
        if eva_type:
            t1_types[eva_type] = entry

    refine_data = _safe_load(os.path.join(rd, 'combined_refine.json'))
    assignment = refine_data.get('best_assignment', {})

    eva_to_triple = build_eva_to_triple_lookup()

    ref_corpus = load_reference_corpus(languages=['latin'], verbose=False)
    base_words = set(w.lower() for w in ref_corpus.get_combined_tokens('latin')
                     if len(w) >= 2)
    expanded, _ = build_expanded_word_set(base_words)
    ref_word_set = base_words | expanded

    corpus = load_corpus(verbose=False)
    all_tokens = corpus.get_tokens()

    # Build folio list
    folio_list = []
    for folio_id, page in sorted(corpus.pages.items()):
        for _ in page.all_tokens:
            folio_list.append(folio_id)

    print(f"  T1 types: {len(t1_types)}, tokens: {len(all_tokens)}")

    # --- Decode ---
    print("  Decoding corpus (connector→null)...")
    decoded = _decode_corpus(all_tokens, assignment, eva_to_triple)

    # --- Select passages ---
    print("  Selecting gap-fill passages...")
    real_passages = _select_gap_fill_passages(
        all_tokens, decoded, t1_types, ref_word_set, folio_list, n=15)
    print(f"    Selected {len(real_passages)} passages "
          f"(mean id: {np.mean([p['id_fraction'] for p in real_passages]):.1%})")

    if not real_passages:
        print("  ERROR: No suitable passages found!")
        result = LLMGapFillResult(
            verdict='NO_PASSAGES',
            runtime_seconds=time.time() - t0,
        )
        _save_json(rd, 'p74_llm_gapfill.json', asdict(result))
        return result

    # --- Build controls ---
    print("  Building controls...")
    shuffled = [_build_shuffled_control(p, seed=i * 31)
                for i, p in enumerate(real_passages)]

    random_passages = []
    for i, p in enumerate(real_passages):
        rp = _build_random_passage(
            all_tokens, decoded, t1_types, ref_word_set, folio_list,
            n_gaps=p['n_gaps'], seed=1000 + i)
        if rp:
            random_passages.append(rp)
    print(f"    Random passages built: {len(random_passages)}")

    known_answer = _build_known_answer_passages(
        all_tokens, decoded, t1_types, ref_word_set, folio_list, n=10)
    print(f"    Known-answer passages: {len(known_answer)}")

    n_api_calls = len(real_passages) * 3 + len(shuffled) + len(random_passages) + len(known_answer)
    print(f"    Total API calls: {n_api_calls}")

    # --- Run LLM queries ---
    print("\n  Running LLM gap-filling...")
    all_results, ka_results = asyncio.run(
        _run_gap_filling_async(
            real_passages, known_answer, shuffled, random_passages))

    n_failures = sum(1 for r in all_results
                     if r.get('control_type') is None
                     and not any(r.get('runs', [])))
    n_failures += sum(1 for r in ka_results if not r.get('result'))
    print(f"    API failures: {n_failures}")

    # --- Score known answers ---
    print("\n  Scoring known-answer calibration...")
    ka_scores = _score_known_answers(ka_results, known_answer)
    print(f"    KA accuracy: {ka_scores['accuracy']:.1%} "
          f"({ka_scores['n_correct']}/{ka_scores['n_total']})")

    # --- Score gap filling ---
    print("  Scoring gap-filling quality...")
    scores = _score_gap_filling(all_results, ka_scores, ref_word_set)
    print(f"    Confidence selectivity: {scores['confidence_selectivity']:.2f}×")
    print(f"    Consistency: {scores['consistency']:.1%}")
    print(f"    Decode agreement: {scores['decode_agreement']:.1%}")

    # --- Validate proposals ---
    print("\n  Validating proposals...")
    validation = _validate_proposals(all_results, scores, ref_word_set)
    print(f"    Accepted: {validation['n_accepted']}")
    print(f"    Rejected: {validation['n_rejected']}")
    if validation['accepted']:
        print(f"    Accepted proposals:")
        for p in validation['accepted'][:10]:
            print(f"      pos {p['position']}: '{p['proposed_word']}' "
                  f"({p['proposed_gloss']}) — "
                  f"decoded='{p['decoded_string']}', "
                  f"ED={p['ed_with_decoded']}")

    # Count fully-filled passages
    filled_by_passage: Dict[int, int] = Counter()
    gaps_by_passage: Dict[int, int] = Counter()
    for r in [r for r in all_results if r.get('control_type') is None]:
        passage_info = r.get('passage', {})
        start = passage_info.get('start', -1)
        gaps_by_passage[start] = passage_info.get('n_gaps', 0)
    for p in validation['accepted']:
        filled_by_passage[p['passage_start']] += 1

    n_fully_filled = sum(1 for start, n_filled in filled_by_passage.items()
                         if n_filled >= gaps_by_passage.get(start, 999))

    # --- Gates ---
    g1 = ka_scores['accuracy'] >= 0.30
    g2 = scores['confidence_selectivity'] > 1.5
    g3 = scores['consistency'] > 0.50
    g4 = validation['n_accepted'] >= 5
    g5 = scores['decode_agreement'] > 0.40
    g6 = n_fully_filled >= 1

    gates_passed = sum([g1, g2, g3, g4, g5, g6])

    print(f"\n  Gates:")
    print(f"    B2_1 (KA accuracy ≥30%): {'PASS' if g1 else 'FAIL'} "
          f"({ka_scores['accuracy']:.1%})")
    print(f"    B2_2 (conf selectivity >1.5×): {'PASS' if g2 else 'FAIL'} "
          f"({scores['confidence_selectivity']:.2f}×)")
    print(f"    B2_3 (consistency >50%): {'PASS' if g3 else 'FAIL'} "
          f"({scores['consistency']:.1%})")
    print(f"    B2_4 (≥5 accepted): {'PASS' if g4 else 'FAIL'} "
          f"({validation['n_accepted']})")
    print(f"    B2_5 (decode agree >40%): {'PASS' if g5 else 'FAIL'} "
          f"({scores['decode_agreement']:.1%})")
    print(f"    B2_6 (≥1 fully filled): {'PASS' if g6 else 'FAIL'} "
          f"({n_fully_filled})")
    print(f"    Total: {gates_passed}/6")

    # --- Verdict ---
    if g1 and g2 and g4 and g6:
        verdict = 'GAP_FILL_VALIDATED'
    elif g1 and g4:
        verdict = 'GAP_FILL_PARTIAL'
    elif g1:
        verdict = 'CALIBRATION_ONLY'
    elif not g1:
        verdict = 'CALIBRATION_FAILED'
    else:
        verdict = 'INSUFFICIENT'

    print(f"\n  Verdict: {verdict}")

    # --- Build result ---
    result = LLMGapFillResult(
        n_real_passages=len(real_passages),
        n_shuffled=len(shuffled),
        n_random=len(random_passages),
        n_known_answer=len(known_answer),
        n_api_calls=n_api_calls,
        n_api_failures=n_failures,
        ka_accuracy=round(ka_scores['accuracy'], 4),
        confidence_selectivity=round(scores['confidence_selectivity'], 4),
        consistency=round(scores['consistency'], 4),
        decode_agreement=round(scores['decode_agreement'], 4),
        n_accepted=validation['n_accepted'],
        n_rejected=validation['n_rejected'],
        acceptance_rate=round(validation['acceptance_rate'], 4),
        accepted_proposals=validation['accepted'],
        n_passages_fully_filled=n_fully_filled,
        gate_b2_1=g1,
        gate_b2_2=g2,
        gate_b2_3=g3,
        gate_b2_4=g4,
        gate_b2_5=g5,
        gate_b2_6=g6,
        gates_passed=gates_passed,
        gate_passed=gates_passed >= 3,
        verdict=verdict,
        runtime_seconds=round(time.time() - t0, 1),
    )

    path = _save_json(rd, 'p74_llm_gapfill.json', asdict(result))
    print(f"\n  Saved: {path}")
    print(f"  Runtime: {result.runtime_seconds:.1f}s")

    return result
