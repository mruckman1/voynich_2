"""
PDF2Text.py — Generic PDF transcription pipeline using Gemini 3.1 Pro via OpenRouter.

Usage:
    python PDF2Text.py --pdf <path_to_pdf> --prompt <prompt_name> [options]

Arguments:
    --pdf         Path to the PDF file to transcribe (required)
    --prompt      Name of the prompt file in the Prompts directory, without extension
    --output      Output JSON path (default: <pdf_stem>_extracted.json next to the PDF)
    --chunk       Pages per API call (default: 2)
                  Dense dictionaries like Cappelli need 1-2 pages per chunk.
                  Plate-based sources like Schmitz can use 3-5.
    --concur      Max concurrent calls (default: 2)
                  Lower = fewer rate-limit errors. Use 1 for guaranteed sequential.
    --delay       Seconds to wait between starting each chunk (default: 1.0)
                  Helps avoid 429 bursts. Use 2-3 if rate limits persist.
    --effort      low | medium | high (default: low)
    --start-page  First page to process, 1-indexed (default: 1)
    --end-page    Last page to process, inclusive (default: last)
    --dpi         Render DPI (default: 150; use 200-300 for manuscripts)
    --resume      Resume from existing checkpoint file

Output format:
    The output JSON is a single object:
      "summary"  — high-level statistics about the run
      "entries"  — the full array of extracted entries

Progress & resuming:
    After every completed chunk results are written to:
        <pdf_stem>_checkpoint.json
    If interrupted, re-run with --resume to continue from where it stopped.
    The checkpoint is only deleted on fully successful completion (0 failed chunks).
    If any chunks failed, the checkpoint is kept so you can --resume.

Prompts directory:
    /Users/mruckman1/Desktop/dev/voynich_2/archive/scripts/Prompts/

Recommended settings by source:
    Cappelli (dense dictionary):
        --chunk 2 --concur 2 --delay 1
    Schmitz / Chatelain (plate images):
        --chunk 3 --concur 2 --delay 1 --dpi 200
    Voynich ligature (manuscript detail):
        --chunk 1 --concur 1 --effort medium --dpi 250
"""

import asyncio
import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from openai import AsyncOpenAI

try:
    import fitz  # pymupdf
except ImportError:
    print("[ERROR] pymupdf is required. Run: uv add pymupdf")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
PROMPTS_DIR = Path("/Users/mruckman1/Desktop/dev/voynich_2/archive/scripts/Prompts")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
MODEL_ID = "google/gemini-3.1-pro-preview"

_checkpoint_lock = threading.Lock()


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pdf",        required=True)
    p.add_argument("--prompt",     required=True)
    p.add_argument("--output",     default=None)
    p.add_argument("--chunk",      type=int,   default=1,
                   help="Pages per API call (default: 1)")
    p.add_argument("--concur",     type=int,   default=2,
                   help="Max concurrent calls (default: 2)")
    p.add_argument("--delay",      type=float, default=1.0,
                   help="Seconds between chunk starts (default: 1.0)")
    p.add_argument("--effort",     default="low", choices=["low", "medium", "high"])
    p.add_argument("--start-page", type=int,   default=1)
    p.add_argument("--end-page",   type=int,   default=None)
    p.add_argument("--dpi",        type=int,   default=150)
    p.add_argument("--resume",     action="store_true")
    return p.parse_args()


# ── Prompt ─────────────────────────────────────────────────────────────────────
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        available = [p.stem for p in sorted(PROMPTS_DIR.glob("*.txt"))]
        print(f"[ERROR] Prompt not found: {path}")
        if available:
            print(f"  Available: {', '.join(available)}")
        sys.exit(1)
    text = path.read_text(encoding="utf-8").strip()
    print(f"  Prompt loaded: {path.name} ({len(text):,} chars)")
    return text


# ── Paths ──────────────────────────────────────────────────────────────────────
def resolve_paths(pdf_path: Path, output_arg):
    base = pdf_path.parent / pdf_path.stem
    output_path     = Path(output_arg) if output_arg else Path(f"{base}_extracted.json")
    checkpoint_path = Path(f"{base}_checkpoint.json")
    return output_path, checkpoint_path


# ── Checkpoint ─────────────────────────────────────────────────────────────────
def load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"  [WARN] Could not read checkpoint: {e}. Starting fresh.")
    return {}


def save_checkpoint(path: Path, completed: dict):
    with _checkpoint_lock:
        path.write_text(
            json.dumps(completed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


# ── Page range & chunks ────────────────────────────────────────────────────────
def get_page_range(pdf_path: Path, start_page: int, end_page):
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    eff_start = max(1, start_page)
    eff_end   = min(end_page, total) if end_page else total
    if eff_start > eff_end:
        print(f"[ERROR] --start-page ({eff_start}) exceeds --end-page ({eff_end}). "
              f"PDF has {total} pages.")
        sys.exit(1)
    print(f"  Total PDF pages:   {total}")
    print(f"  Processing pages:  {eff_start}–{eff_end} ({eff_end - eff_start + 1} pages)")
    return eff_start, eff_end, total


def build_chunks(eff_start: int, eff_end: int, pages_per_chunk: int):
    chunks = []
    for s in range(eff_start, eff_end + 1, pages_per_chunk):
        e = min(s + pages_per_chunk - 1, eff_end)
        chunks.append((s, e))
    return chunks


# ── Rendering ──────────────────────────────────────────────────────────────────
def render_page_to_base64(pdf_path: Path, page_index_0: int, dpi: int) -> str:
    doc = fitz.open(str(pdf_path))
    page = doc[page_index_0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


# ── API call with explicit 429 retry ──────────────────────────────────────────
async def transcribe_chunk(
    client: AsyncOpenAI,
    pdf_path: Path,
    start_page: int,
    end_page: int,
    prompt: str,
    effort: str,
    dpi: int,
    max_retries: int = 6,
) -> tuple[int, str | None]:
    content = []
    for page_num in range(start_page, end_page + 1):
        try:
            b64 = render_page_to_base64(pdf_path, page_num - 1, dpi)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except Exception as e:
            print(f"  [WARN] Could not render page {page_num}: {e}")

    if not content:
        return start_page, None

    content.append({"type": "text", "text": prompt})

    backoff = 5.0  # initial backoff in seconds for 429s
    # Timeout: 180s for complex manuscript pages, 90s for typeset sources.
    # Handwritten cipher pages (Fontana) can take 60-120s; typeset (Cappelli) ~10-20s.
    timeout_seconds = 180.0
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": content}],
                    reasoning_effort=effort,
                ),
                timeout=timeout_seconds,
            )
            return start_page, response.choices[0].message.content

        except asyncio.TimeoutError:
            is_last = attempt == max_retries - 1
            if is_last:
                print(f"  [TIMEOUT] Pages {start_page}–{end_page}: gave up after "
                      f"{max_retries} attempts ({timeout_seconds:.0f}s each).")
                return start_page, None
            print(f"  [TIMEOUT] Pages {start_page}–{end_page}: "
                  f"no response after {timeout_seconds:.0f}s, retrying "
                  f"(attempt {attempt + 1}/{max_retries})...")
            await asyncio.sleep(5.0)

        except Exception as e:
            err_str = str(e)
            is_rate_limit   = "429" in err_str or "rate" in err_str.lower()
            is_server_error = any(code in err_str for code in ("502", "503", "504", "500"))
            is_last         = attempt == max_retries - 1

            if is_last:
                print(f"  [ERROR] Pages {start_page}–{end_page}: gave up after "
                      f"{max_retries} attempts. Last error: {e}")
                return start_page, None

            if is_rate_limit or is_server_error:
                wait = backoff * (2 ** attempt)  # exponential: 5, 10, 20, 40, 80...
                label = "RATE LIMIT" if is_rate_limit else "SERVER ERROR"
                print(f"  [{label}] Pages {start_page}–{end_page}: "
                      f"waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait)
            else:
                # Other error — short wait then retry
                print(f"  [ERROR] Pages {start_page}–{end_page}: {e} "
                      f"(attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(3.0)

    return start_page, None  # unreachable but satisfies type checker


# ── JSON parsing ───────────────────────────────────────────────────────────────
def parse_chunk_json(
    raw_text: str | None,
    start_page: int,
    end_page: int,
    output_path: Path,
) -> tuple[list[dict], bool]:
    """
    Returns (entries, success).
    success=False means the API call itself failed (None response),
    not that the page had no entries (empty list is a valid result for
    front matter / appendix pages).
    """
    if raw_text is None:
        return [], False  # API failure — should be retried

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()

    # Handle empty response (model returned nothing for a skippable page)
    if not cleaned or cleaned in ("[]", "[ ]"):
        return [], True  # valid empty result, not a failure

    try:
        entries = json.loads(cleaned)
        if not isinstance(entries, list):
            print(f"  [WARN] Pages {start_page}–{end_page}: not a JSON array — "
                  f"got {type(entries).__name__}")
            raw_path = (
                output_path.parent
                / f"{output_path.stem}_pages_{start_page}_{end_page}.raw.txt"
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            return [], False
        return entries, True
    except json.JSONDecodeError as e:
        print(f"  [WARN] Pages {start_page}–{end_page}: JSON parse failed — {e}")
        raw_path = (
            output_path.parent
            / f"{output_path.stem}_pages_{start_page}_{end_page}.raw.txt"
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        print(f"           Raw saved → {raw_path.name}")
        return [], False


# ── Summary builder ────────────────────────────────────────────────────────────
def build_summary(
    entries: list[dict],
    args,
    pdf_path: Path,
    total_pdf_pages: int,
    eff_start: int,
    eff_end: int,
    all_chunks: list,
    completed_chunks: int,
    failed_chunks: int,
    started_at: str,
) -> dict:
    n = len(entries)

    summary = {
        "run": {
            "started_at":       started_at,
            "completed_at":     datetime.now(timezone.utc).isoformat(),
            "pdf_file":         pdf_path.name,
            "prompt":           args.prompt,
            "model":            MODEL_ID,
            "effort":           args.effort,
            "dpi":              args.dpi,
            "chunk_size":       args.chunk,
            "concurrency":      args.concur,
            "delay_seconds":    args.delay,
            "pages_in_pdf":     total_pdf_pages,
            "pages_processed":  f"{eff_start}–{eff_end}",
            "total_chunks":     len(all_chunks),
            "completed_chunks": completed_chunks,
            "failed_chunks":    failed_chunks,
        },
        "totals": {
            "entries_total":          n,
            "entries_high_conf":      sum(1 for e in entries if e.get("confidence") == "high"),
            "entries_medium_conf":    sum(1 for e in entries if e.get("confidence") == "medium"),
            "entries_low_conf":       sum(1 for e in entries if e.get("confidence") == "low"),
            "entries_with_null_form": sum(1 for e in entries if e.get("abbreviated_form") is None),
            "entries_with_null_exp":  sum(1 for e in entries if e.get("latin_expansion") is None),
        },
    }

    all_keys = {k for e in entries for k in e}

    if "has_special_symbols" in all_keys:
        summary["totals"]["entries_with_special_symbols"] = sum(
            1 for e in entries if e.get("has_special_symbols") is True
        )
        summary["totals"]["entries_with_visual_description"] = sum(
            1 for e in entries if e.get("visual_description") is not None
        )

    if "priority" in all_keys:
        summary["totals"]["entries_priority_high"] = sum(
            1 for e in entries if e.get("priority") == "high"
        )

    if "language" in all_keys:
        summary["language_breakdown"] = dict(
            sorted(Counter(e.get("language") or "missing" for e in entries).items())
        )

    if "semantic_domain" in all_keys:
        summary["semantic_domain_breakdown"] = dict(
            sorted(
                Counter(e.get("semantic_domain") or "missing" for e in entries).items(),
                key=lambda x: -x[1]
            )
        )

    if "provenance" in all_keys:
        prov_counts = Counter(
            e.get("provenance") for e in entries if e.get("provenance") is not None
        )
        summary["provenance_breakdown"] = dict(
            sorted(prov_counts.items(), key=lambda x: -x[1])
        )

    if "century_ref" in all_keys:
        def normalise_century(c):
            if c is None:
                return "unknown"
            c = c.upper().strip().rstrip(".")
            for roman in ["XVII", "XVI", "XV", "XIV", "XIII", "XII", "XI", "X",
                          "IX", "VIII", "VII", "VI"]:
                if roman in c:
                    return roman
            return c[:10]

        century_counts = Counter(normalise_century(e.get("century_ref")) for e in entries)
        summary["century_breakdown"] = dict(
            sorted(century_counts.items(), key=lambda x: -x[1])
        )
        target_centuries = {"XIII", "XIV", "XV"}
        summary["totals"]["entries_voynich_period"] = sum(
            1 for e in entries
            if normalise_century(e.get("century_ref")) in target_centuries
        )

    return summary


# ── Terminal diagnostics ───────────────────────────────────────────────────────
def print_diagnostics(summary: dict, output_path: Path, lc_path: Path | None):
    t = summary["totals"]
    r = summary["run"]

    print(f"\n{'─' * 56}")
    print(f"  Done.  {r['completed_at']}")
    print(f"\n  Entries extracted:        {t['entries_total']:>6}")
    print(f"  High confidence:          {t['entries_high_conf']:>6}")
    if t.get("entries_medium_conf", 0) > 0:
        print(f"  Medium confidence:        {t['entries_medium_conf']:>6}")
    if t.get("entries_low_conf", 0) > 0:
        print(f"  Low confidence:           {t['entries_low_conf']:>6}")
    if t.get("entries_with_null_form", 0) > 0:
        print(f"  Null abbreviated_form:    {t['entries_with_null_form']:>6}")
    if t.get("entries_with_null_exp", 0) > 0:
        print(f"  Null latin_expansion:     {t['entries_with_null_exp']:>6}")

    if "entries_with_special_symbols" in t:
        print(f"\n  Special symbols:          {t['entries_with_special_symbols']:>6}")
        print(f"  Visual descriptions:      {t.get('entries_with_visual_description', 0):>6}")
    if "entries_priority_high" in t:
        print(f"  Priority HIGH:            {t['entries_priority_high']:>6}")
    if "entries_voynich_period" in t:
        print(f"  Voynich period (XIII–XV): {t['entries_voynich_period']:>6}")

    if "language_breakdown" in summary:
        print(f"\n  Language:")
        for lang, count in summary["language_breakdown"].items():
            print(f"    {lang:<14} {count:>5}")

    if "semantic_domain_breakdown" in summary:
        print(f"\n  Semantic domains:")
        for dom, count in summary["semantic_domain_breakdown"].items():
            print(f"    {str(dom):<16} {count:>5}")

    if "provenance_breakdown" in summary and summary["provenance_breakdown"]:
        print(f"\n  Provenance tags (top 10):")
        for prov, count in list(summary["provenance_breakdown"].items())[:10]:
            print(f"    {str(prov):<24} {count:>5}")

    if r.get("failed_chunks", 0) > 0:
        print(f"\n  [WARN] {r['failed_chunks']} chunk(s) failed — checkpoint kept.")
        print(f"         Re-run with --resume to retry failed chunks.")

    print(f"\n  Output:  {output_path}")
    if lc_path:
        print(f"  Flagged: {lc_path}")
    print(f"{'─' * 56}")


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_KEY_HERE":
        print("[ERROR] OPENROUTER_API_KEY is not set in the script.")
        sys.exit(1)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)

    output_path, checkpoint_path = resolve_paths(pdf_path, args.output)
    prompt = load_prompt(args.prompt)

    print(f"\nPDF2Text — {pdf_path.name}")
    print(f"  Prompt:      {args.prompt}")
    print(f"  Output:      {output_path}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Chunk size:  {args.chunk} pages")
    print(f"  Concurrency: {args.concur}")
    print(f"  Delay:       {args.delay}s between chunks")
    print(f"  Effort:      {args.effort}")
    print(f"  Render DPI:  {args.dpi}")
    if args.start_page > 1 or args.end_page:
        print(f"  Page range:  {args.start_page}–{args.end_page or 'end'}")
    print()

    eff_start, eff_end, total_pdf_pages = get_page_range(
        pdf_path, args.start_page, args.end_page
    )
    all_chunks = build_chunks(eff_start, eff_end, args.chunk)

    # ── Resume ─────────────────────────────────────────────────────────────────
    completed: dict[int, list[dict]] = {}

    if args.resume and checkpoint_path.exists():
        completed = load_checkpoint(checkpoint_path)
        print(f"  Resuming: {len(completed)} chunks done, "
              f"{len(all_chunks) - len(completed)} remaining.\n")
    elif not args.resume and checkpoint_path.exists():
        print(f"  [NOTE] Checkpoint exists — run with --resume to continue.")
        print(f"         Starting fresh (checkpoint will be overwritten).\n")

    # Track which chunks truly failed (API returned None)
    failed_chunk_starts: set[int] = set()

    pending = [(s, e) for s, e in all_chunks if s not in completed]
    print(f"  Total chunks: {len(all_chunks)}  |  Pending: {len(pending)}\n")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_retries=0,  # We handle retries ourselves in transcribe_chunk
    )
    semaphore = asyncio.Semaphore(args.concur)

    # Inter-chunk delay: a shared lock ensures each chunk waits --delay seconds
    # after the previous one starts, regardless of total chunk count.
    # This is a fixed gap (e.g. 2s between starts), not a cumulative stagger.
    last_start_time = [0.0]  # mutable so the closure can write to it
    delay_lock = asyncio.Lock()

    async def process_chunk(s, e, chunk_index: int):
        async with delay_lock:
            # Enforce minimum gap between chunk starts
            now = asyncio.get_event_loop().time()
            gap = now - last_start_time[0]
            if gap < args.delay and chunk_index > 0:
                await asyncio.sleep(args.delay - gap)
            last_start_time[0] = asyncio.get_event_loop().time()
        async with semaphore:
            print(f"  → Pages {s}–{e}...")
            start_page, raw_text = await transcribe_chunk(
                client, pdf_path, s, e, prompt, args.effort, args.dpi
            )
            entries, success = parse_chunk_json(raw_text, s, e, output_path)

            if not success:
                failed_chunk_starts.add(start_page)

            completed[start_page] = entries
            save_checkpoint(checkpoint_path, completed)

            done = len(completed)
            total = len(all_chunks)
            status = "✓" if success else "✗"
            print(f"     {status} Pages {s}–{e}  "
                  f"[{done}/{total}  {100*done//total}%]  "
                  f"{len(entries)} entries")

    await asyncio.gather(*[
        process_chunk(s, e, i) for i, (s, e) in enumerate(pending)
    ])

    # ── Merge in page order ────────────────────────────────────────────────────
    all_entries: list[dict] = []
    low_confidence: list[dict] = []

    for s, _ in all_chunks:
        chunk_entries = completed.get(s, [])
        for entry in chunk_entries:
            if entry.get("confidence") == "low":
                low_confidence.append(entry)
        all_entries.extend(chunk_entries)

    failed_chunks = len(failed_chunk_starts)

    # ── Build and write output ─────────────────────────────────────────────────
    summary = build_summary(
        entries=all_entries,
        args=args,
        pdf_path=pdf_path,
        total_pdf_pages=total_pdf_pages,
        eff_start=eff_start,
        eff_end=eff_end,
        all_chunks=all_chunks,
        completed_chunks=len(completed),
        failed_chunks=failed_chunks,
        started_at=started_at,
    )

    output = {"summary": summary, "entries": all_entries}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lc_path = None
    if low_confidence:
        lc_path = output_path.parent / f"{output_path.stem}_low_confidence.json"
        lc_path.write_text(
            json.dumps(low_confidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print_diagnostics(summary, output_path, lc_path)

    # Only delete checkpoint if every chunk succeeded
    if failed_chunks == 0:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"  Checkpoint deleted (all chunks succeeded).")
    else:
        print(f"  Checkpoint kept — re-run with --resume to retry "
              f"{failed_chunks} failed chunk(s).")


if __name__ == "__main__":
    asyncio.run(main())