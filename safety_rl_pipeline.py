"""
Safety RL Training Data Pipeline
=================================
Pulls and formats 4 key safety datasets into DPO preference pairs
for reinforcement learning / LLM-as-judge training.

Datasets:
  1. R-Judge       - Agent safety risk awareness (GitHub)
  2. PKU-SafeRLHF  - Dual helpfulness/harmlessness preferences (HuggingFace)
  3. HarmBench     - Red teaming attack behaviors (GitHub)
  4. BeaverTails   - Safety-labeled QA pairs (HuggingFace)

Output schema (JSONL):
  {
    "id": str,
    "source": str,
    "prompt": str,
    "chosen": str,       # safe / preferred response
    "rejected": str,     # unsafe / dispreferred response
    "metadata": {
      "risk_type": str,
      "harm_category": str,
      "severity": str,   # low | moderate | severe
      "safety_label": bool,
      "turns": int       # for multi-turn records
    }
  }

Usage:
  pip install datasets huggingface_hub requests tqdm

  # Standard (downloads HF datasets to ~/.cache/huggingface):
  python safety_rl_pipeline.py --datasets all --output ./rl_safety_data --samples 1000

  # Streaming — no HuggingFace cache download, output JSONL written locally:
  python safety_rl_pipeline.py --datasets all --output ./rl_safety_data --stream

  # Fully virtual — nothing touches local disk at all:
  python safety_rl_pipeline.py --datasets all --stream --no-output

  # Pipe directly into TRL DPO training (fully virtual):
  python safety_rl_pipeline.py --datasets all --stream --no-output --format trl | \\
      python train_safety_dpo.py --stdin

  # With HuggingFace auth token (higher rate limits, private datasets):
  python safety_rl_pipeline.py --datasets all --stream --hf-token hf_xxxx
"""

import argparse
import json
import os
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Generator, Iterator, Optional

# ── Optional imports (fail gracefully) ────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def make_record(
    source: str,
    prompt: str,
    chosen: str,
    rejected: str,
    risk_type: str = "",
    harm_category: str = "",
    severity: str = "",
    safety_label: bool = True,
    turns: int = 1,
) -> dict:
    """Canonical DPO preference record."""
    return {
        "id": str(uuid.uuid4()),
        "source": source,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "risk_type": risk_type,
            "harm_category": harm_category,
            "severity": severity,
            "safety_label": safety_label,
            "turns": turns,
        },
    }


def to_trl_format(record: dict) -> dict:
    """
    Strip to minimal TRL DPOTrainer schema.
    Extra keys are harmless to TRL but this keeps output lean.
    """
    return {
        "prompt":        record["prompt"],
        "chosen":        record["chosen"],
        "rejected":      record["rejected"],
        "harm_category": record["metadata"]["harm_category"],
        "severity":      record["metadata"]["severity"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — R-JUDGE
# Always streams (small GitHub JSON files fetched via requests, no local cache)
# ═══════════════════════════════════════════════════════════════════════════════

RJUDGE_BASE  = "https://raw.githubusercontent.com/Lordog/R-Judge/main/data/"
RJUDGE_FILES = ["train.json", "test.json"]


def _rjudge_turns_to_prompt(record: dict) -> str:
    lines = [f"User instruction: {record.get('instruction', '')}"]
    for i, turn in enumerate(record.get("interaction", []), 1):
        action = turn.get("action", turn.get("assistant", ""))
        obs    = turn.get("observation", turn.get("feedback", ""))
        lines.append(f"[Turn {i}] Agent action: {action}")
        if obs:
            lines.append(f"[Turn {i}] Environment: {obs}")
    return "\n".join(lines)


def _rjudge_safe_response(record: dict) -> str:
    label = record.get("label", "unsafe")
    desc  = record.get("risk_description", "No specific risk identified.")
    if label == "safe":
        return f"SAFE. The agent interaction does not present a safety risk. {desc}"
    return (
        f"UNSAFE. Risk identified: {desc} "
        f"The agent should have refused or flagged this action."
    )


def fetch_rjudge(
    max_records: Optional[int] = None,
    stream: bool = False,       # accepted for API consistency; R-Judge always streams
    hf_token: Optional[str] = None,
) -> Generator[dict, None, None]:
    """Yield DPO records from R-Judge (always streamed via requests — no disk cache)."""
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    count = 0
    for fname in RJUDGE_FILES:
        url  = RJUDGE_BASE + fname
        print(f"  Streaming R-Judge: {url}", file=sys.stderr)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data if isinstance(data, list) else data.get("data", [])
        for rec in records:
            if max_records and count >= max_records:
                return
            label = rec.get("label", "unsafe").lower()
            yield make_record(
                source="r_judge",
                prompt=_rjudge_turns_to_prompt(rec),
                chosen=_rjudge_safe_response(rec),
                rejected=(
                    "The agent completed the task as instructed. "
                    "No safety concerns are flagged."
                ),
                risk_type=rec.get("risk_type", ""),
                harm_category=rec.get("application_category", ""),
                severity="moderate",
                safety_label=(label == "unsafe"),
                turns=len(rec.get("interaction", [])),
            )
            count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — PKU-SafeRLHF
# Each row already contains both responses — streaming is trivial
# ═══════════════════════════════════════════════════════════════════════════════

PKU_HARM_CATEGORIES = [
    "Animal Abuse", "Child Abuse", "Controversial Topics, Politics",
    "Discrimination, Stereotype, Injustice", "Drug Abuse, Weapons, Banned Substance",
    "Financial Crime, Property Crime, Theft", "Hate Speech, Offensive Language",
    "Human Trafficking", "Non-Violent Unethical Behavior", "Privacy Violation",
    "Self-Harm", "Sex-Related Crimes", "Sexual Content",
    "Terrorism, Organized Crime", "Violence, Aiding and Abetting, Incitement",
    "Endangering National Security", "Cyber Crime", "Psychological Manipulation",
    "Other Illegal Activity",
]


def _pku_severity(row: dict) -> str:
    lvl = row.get("severity_level", row.get("severity", 0))
    return {1: "low", 2: "moderate", 3: "severe"}.get(int(lvl or 0), "unknown")


def _pku_harm_cats(row: dict) -> str:
    if row.get("category"):
        return row["category"]
    cats = []
    for cat in PKU_HARM_CATEGORIES:
        col = cat.lower().replace(" ", "_").replace(",", "").replace("-", "_")
        if row.get(col) or row.get(f"response_0_{col}") or row.get(f"response_1_{col}"):
            cats.append(cat)
    return "; ".join(cats) if cats else "general_harm"


def fetch_pku_saferlhf(
    split: str = "train",
    max_records: Optional[int] = None,
    stream: bool = False,
    hf_token: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Yield DPO records from PKU-SafeRLHF.

    stream=True  → HuggingFace IterableDataset, no local cache written.
                   Each row has both responses already — pairing is trivial.
    stream=False → Downloads and caches dataset locally (~400 MB).
    """
    if not HAS_DATASETS:
        raise ImportError("pip install datasets huggingface_hub")

    mode = "streaming" if stream else "cached download"
    print(f"  Loading PKU-SafeRLHF ({split}, {mode})…", file=sys.stderr)

    ds = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF",
        split=split,
        streaming=stream,
        token=hf_token,
        trust_remote_code=True,
    )

    count = 0
    for row in tqdm(ds, desc="PKU-SafeRLHF", file=sys.stderr):
        if max_records and count >= max_records:
            break
        r0_safe = bool(row.get("is_response_0_safe", True))
        r1_safe = bool(row.get("is_response_1_safe", True))
        if r0_safe == r1_safe:
            continue   # no contrast → skip

        if r0_safe:
            chosen, rejected = row["response_0"], row["response_1"]
        else:
            chosen, rejected = row["response_1"], row["response_0"]

        yield make_record(
            source="pku_saferlhf",
            prompt=row["prompt"],
            chosen=chosen,
            rejected=rejected,
            risk_type="policy_violation",
            harm_category=_pku_harm_cats(row),
            severity=_pku_severity(row),
            safety_label=True,
            turns=1,
        )
        count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 3 — HarmBench
# Always streams (small GitHub CSV fetched via requests, no local cache)
# ═══════════════════════════════════════════════════════════════════════════════

HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

HARMBENCH_REFUSALS = {
    "standard": (
        "I'm not able to help with that request. It involves content that could "
        "cause harm. If you have a legitimate need, please consult appropriate "
        "professional resources or authorities."
    ),
    "cybercrime": (
        "I can't assist with creating malware, exploits, or unauthorized access tools. "
        "For legitimate security research, please use controlled environments and "
        "follow responsible disclosure practices."
    ),
    "chemical_biological": (
        "I'm unable to provide instructions for creating dangerous substances or weapons. "
        "This type of information poses serious safety risks."
    ),
    "harassment": (
        "I can't generate content designed to harass, threaten, or intimidate individuals. "
        "If you're experiencing a conflict, consider reaching out to appropriate support."
    ),
}


def _harmbench_refusal(category: str) -> str:
    cat = category.lower()
    if "cyber" in cat or "malware" in cat or "hack" in cat:
        return HARMBENCH_REFUSALS["cybercrime"]
    if "chemical" in cat or "bio" in cat or "weapon" in cat:
        return HARMBENCH_REFUSALS["chemical_biological"]
    if "harass" in cat or "threaten" in cat:
        return HARMBENCH_REFUSALS["harassment"]
    return HARMBENCH_REFUSALS["standard"]


def fetch_harmbench(
    max_records: Optional[int] = None,
    stream: bool = False,       # accepted for API consistency; HarmBench always streams
    hf_token: Optional[str] = None,
) -> Generator[dict, None, None]:
    """Yield DPO records from HarmBench (always streamed via requests — no disk cache)."""
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    import csv, io
    print(f"  Streaming HarmBench: {HARMBENCH_CSV}", file=sys.stderr)
    resp = requests.get(HARMBENCH_CSV, timeout=30)
    resp.raise_for_status()

    count = 0
    for row in csv.DictReader(io.StringIO(resp.text)):
        if max_records and count >= max_records:
            break
        behavior  = row.get("Behavior", row.get("behavior", ""))
        category  = row.get("SemanticCategory", row.get("category", "general_harm"))
        context   = row.get("ContextString", "")
        prompt    = f"Context: {context}\n\nRequest: {behavior}" if context else behavior

        yield make_record(
            source="harmbench",
            prompt=prompt,
            chosen=_harmbench_refusal(category),
            rejected=(
                f"Sure, here's how to {behavior.lower().rstrip('.')}. "
                f"[HARMFUL CONTENT — used only as rejected signal for RL training]"
            ),
            risk_type="red_team_attack",
            harm_category=category,
            severity="severe",
            safety_label=True,
            turns=1,
        )
        count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 4 — BeaverTails
# Pairing strategy differs between stream and non-stream modes
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_beavertails(
    split: str = "30k_train",
    max_records: Optional[int] = None,
    stream: bool = False,
    hf_token: Optional[str] = None,
    stream_buffer: int = 5_000,
) -> Generator[dict, None, None]:
    """
    Yield DPO records from BeaverTails.

    BeaverTails has one response per row, so we must pair safe/unsafe responses
    for the same prompt. Two strategies:

    stream=False — Full scan: load all rows into a dict, then yield matched pairs.
                   Best pair-matching rate (~95%+). Uses ~200 MB RAM peak.

    stream=True  — Bounded buffer: hold up to `stream_buffer` unmatched rows in an
                   OrderedDict. When a match is found, yield immediately and evict
                   both entries. Oldest unmatched entries are dropped when buffer
                   is full. No disk write; ~50 MB RAM peak.
                   Trade-off: ~70–80% pair-match rate (some pairs missed if the
                   matching row is too far ahead in the stream).
    """
    if not HAS_DATASETS:
        raise ImportError("pip install datasets huggingface_hub")

    mode = "streaming + bounded buffer" if stream else "cached download"
    print(f"  Loading BeaverTails ({split}, {mode})…", file=sys.stderr)

    ds = load_dataset(
        "PKU-Alignment/BeaverTails",
        split=split,
        streaming=stream,
        token=hf_token,
        trust_remote_code=True,
    )

    def _make_bt_record(prompt, safe_resp, unsafe_resp, cats):
        return make_record(
            source="beavertails",
            prompt=prompt,
            chosen=safe_resp,
            rejected=unsafe_resp,
            risk_type="policy_violation",
            harm_category="; ".join(cats) if cats else "general_harm",
            severity="moderate",
            safety_label=True,
            turns=1,
        )

    count = 0

    if not stream:
        # ── Non-streaming: full scan into dict, then yield pairs ──────────────
        prompt_map: dict = {}
        for row in tqdm(ds, desc="BeaverTails indexing", file=sys.stderr):
            p       = row["prompt"]
            is_safe = bool(row.get("is_safe", True))
            resp    = row["response"]
            cats    = [k for k, v in row.get("category", {}).items() if v]
            if p not in prompt_map:
                prompt_map[p] = {"safe": None, "unsafe": None, "cats": cats}
            if is_safe and not prompt_map[p]["safe"]:
                prompt_map[p]["safe"] = resp
            elif not is_safe and not prompt_map[p]["unsafe"]:
                prompt_map[p]["unsafe"] = resp
                prompt_map[p]["cats"]   = cats

        for prompt, pair in prompt_map.items():
            if max_records and count >= max_records:
                break
            if not pair["safe"] or not pair["unsafe"]:
                continue
            yield _make_bt_record(prompt, pair["safe"], pair["unsafe"], pair["cats"])
            count += 1

    else:
        # ── Streaming: bounded OrderedDict buffer ─────────────────────────────
        # Key: prompt  Value: {"safe": str|None, "unsafe": str|None, "cats": list}
        buffer: OrderedDict = OrderedDict()

        for row in tqdm(ds, desc="BeaverTails streaming", file=sys.stderr):
            if max_records and count >= max_records:
                break

            p       = row["prompt"]
            is_safe = bool(row.get("is_safe", True))
            resp    = row["response"]
            cats    = [k for k, v in row.get("category", {}).items() if v]

            if p in buffer:
                entry = buffer[p]
                if is_safe and not entry["safe"]:
                    entry["safe"] = resp
                elif not is_safe and not entry["unsafe"]:
                    entry["unsafe"] = resp
                    entry["cats"]   = cats

                # Check if pair is now complete
                if entry["safe"] and entry["unsafe"]:
                    yield _make_bt_record(p, entry["safe"], entry["unsafe"], entry["cats"])
                    del buffer[p]
                    count += 1
            else:
                # Evict oldest entry if buffer is full
                if len(buffer) >= stream_buffer:
                    buffer.popitem(last=False)

                buffer[p] = {
                    "safe":   resp if is_safe else None,
                    "unsafe": None if is_safe else resp,
                    "cats":   cats,
                }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

FETCHERS = {
    "r_judge":      fetch_rjudge,
    "pku_saferlhf": fetch_pku_saferlhf,
    "harmbench":    fetch_harmbench,
    "beavertails":  fetch_beavertails,
}


def run_pipeline(
    datasets: list,
    output_dir: Optional[str],
    samples_per_dataset: Optional[int] = None,
    combined: bool = True,
    stream: bool = False,
    hf_token: Optional[str] = None,
    fmt: str = "full",
) -> dict:
    """
    Pull each dataset, convert to DPO format.

    output_dir=None  → --no-output mode: records yielded/printed only, nothing written.
    stream=True      → HuggingFace streaming (no HF cache), R-Judge/HarmBench unaffected.
    fmt="trl"        → Strip to minimal TRL DPOTrainer schema before output.

    Returns dict of {dataset_name: record_count}.
    """
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    formatter = to_trl_format if fmt == "trl" else (lambda r: r)
    stats = {}
    combined_file = None
    combined_fh   = None

    if out_path and combined:
        combined_file = out_path / "combined_safety_dpo.jsonl"
        combined_fh   = open(combined_file, "w")

    for name in datasets:
        if name not in FETCHERS:
            print(f"⚠️  Unknown dataset: {name}. Skipping.", file=sys.stderr)
            continue

        print(f"\n{'─'*60}", file=sys.stderr)
        print(f"Processing: {name.upper()}  [{'stream' if stream else 'download'}]", file=sys.stderr)
        print(f"{'─'*60}", file=sys.stderr)

        per_file_fh = None
        if out_path:
            per_file = out_path / f"{name}_dpo.jsonl"
            per_file_fh = open(per_file, "w")

        count = 0
        try:
            for rec in FETCHERS[name](
                max_records=samples_per_dataset,
                stream=stream,
                hf_token=hf_token,
            ):
                formatted = formatter(rec)
                line = json.dumps(formatted)

                if per_file_fh:
                    per_file_fh.write(line + "\n")
                if combined_fh:
                    combined_fh.write(line + "\n")
                # --no-output + --stream → write to stdout for piping
                if not out_path:
                    print(line)

                count += 1

        except Exception as e:
            print(f"  ❌ Error fetching {name}: {e}", file=sys.stderr)
        finally:
            if per_file_fh:
                per_file_fh.close()

        stats[name] = count
        dest = str(per_file) if out_path else "stdout"
        print(f"  ✅ {count:,} records → {dest}", file=sys.stderr)

    if combined_fh:
        combined_fh.close()
        print(f"\n{'═'*60}", file=sys.stderr)
        total = sum(stats.values())
        print(f"Combined: {total:,} total records → {combined_file}", file=sys.stderr)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Safety RL Training Data Pipeline — DPO preference pair formatter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and cache HuggingFace datasets locally:
  python safety_rl_pipeline.py --datasets all --output ./rl_safety_data

  # Stream only — no HuggingFace cache, write output JSONL:
  python safety_rl_pipeline.py --datasets all --output ./rl_safety_data --stream

  # Fully virtual — no disk writes at all:
  python safety_rl_pipeline.py --datasets all --stream --no-output

  # Pipe directly to TRL training script:
  python safety_rl_pipeline.py --datasets all --stream --no-output --format trl | \\
      python train_safety_dpo.py --stdin

  # Authenticated access (higher HF rate limits):
  python safety_rl_pipeline.py --datasets all --stream --hf-token hf_xxxx
        """
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=["all", "r_judge", "pku_saferlhf", "harmbench", "beavertails"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--output", default="./rl_safety_data",
        help="Output directory for JSONL files (omit with --no-output)",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Max records per dataset (default: all)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream HuggingFace datasets — no local cache downloaded",
    )
    parser.add_argument(
        "--no-output", action="store_true",
        help="Skip writing any files — stream records to stdout for piping",
    )
    parser.add_argument(
        "--no-combined", action="store_true",
        help="Skip writing the combined JSONL file",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace API token for authenticated access (higher rate limits)",
    )
    parser.add_argument(
        "--format", default="full", choices=["full", "trl"],
        help="Output schema: 'full' (all fields) or 'trl' (minimal TRL DPOTrainer schema)",
    )
    parser.add_argument(
        "--stream-buffer", type=int, default=5_000, metavar="N",
        help="BeaverTails streaming pair-matching buffer size (default: 5000)",
    )

    args = parser.parse_args()

    selected = list(FETCHERS.keys()) if "all" in args.datasets else args.datasets
    output   = None if args.no_output else args.output

    # ── Print header to stderr (won't corrupt stdout pipe) ───────────────────
    print("╔══════════════════════════════════════════════════════╗", file=sys.stderr)
    print("║    Safety RL Training Data Pipeline                  ║", file=sys.stderr)
    print("║    Output format: DPO preference pairs (JSONL)       ║", file=sys.stderr)
    print("╚══════════════════════════════════════════════════════╝", file=sys.stderr)
    print(f"\nDatasets  : {', '.join(selected)}",           file=sys.stderr)
    print(f"Mode      : {'stream (no HF cache)' if args.stream else 'download + cache'}",
          file=sys.stderr)
    print(f"Output    : {'stdout (no disk write)' if args.no_output else args.output}",
          file=sys.stderr)
    print(f"Format    : {args.format}",                     file=sys.stderr)
    print(f"Samples   : {args.samples or 'ALL'}",           file=sys.stderr)
    print(f"HF token  : {'set' if args.hf_token else 'not set (anonymous)'}",
          file=sys.stderr)

    stats = run_pipeline(
        datasets=selected,
        output_dir=output,
        samples_per_dataset=args.samples,
        combined=not args.no_combined,
        stream=args.stream,
        hf_token=args.hf_token,
        fmt=args.format,
    )

    print("\n\n📊 Summary", file=sys.stderr)
    print("─" * 40,       file=sys.stderr)
    total = 0
    for ds, cnt in stats.items():
        print(f"  {ds:<20} {cnt:>6,} records", file=sys.stderr)
        total += cnt
    print("─" * 40, file=sys.stderr)
    print(f"  {'TOTAL':<20} {total:>6,} records\n", file=sys.stderr)


if __name__ == "__main__":
    main()
