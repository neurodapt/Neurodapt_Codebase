"""Prepare the Narrative Memory dataset for MemoryRanker evaluation.

The downloaded repository stores each experiment as a pandas pickle.  A
compiled story contains, among other fields:

    narrative       full story text
    segmentation    newline-separated clause segmentation
    recall scores    one list of recalled clause indices per participant

This script converts those files into a stable JSON format that eval.py can
consume without depending on the original repository's notebooks.

Run from mem_ranker with:

    python eval_pipeline/pre-processing.py

The output is written to eval_pipeline/narrative_memory_eval.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "llm-narrative-analysis" / "data" / "compiled_data"
DEFAULT_OUTPUT = HERE / "narrative_memory_eval.json"


def load_pickle(path: Path) -> Any:
    """Load a pandas pickle while keeping this script usable with pandas data."""
    try:
        import pandas as pd

        return pd.read_pickle(path)
    except (ImportError, ModuleNotFoundError, AttributeError):
        # This fallback is useful when the pickle contains only ordinary
        # Python objects and pandas is unavailable or has changed internals.
        with path.open("rb") as handle:
            return pickle.load(handle)


def get_field(record: Any, *names: str) -> Any:
    """Read a field from a dict-like or pandas Series-like object."""
    for name in names:
        if isinstance(record, dict) and name in record:
            return record[name]
        try:
            return record[name]
        except (KeyError, IndexError, TypeError):
            continue
    raise KeyError(f"None of these fields were found: {names}")


def clean_clauses(segmentation: Any) -> list[str]:
    """Convert newline-separated segmentation into ordered clause text."""
    if isinstance(segmentation, str):
        raw_clauses = segmentation.splitlines()
    else:
        raw_clauses = list(segmentation)

    clauses = []
    for clause in raw_clauses:
        text = str(clause).strip()
        # Some exported segmentations include an initial clause number.
        text = re.sub(r"^\s*\d+\s*[:.)-]\s*", "", text)
        if text:
            clauses.append(text)
    return clauses


def normalize_recall_scores(recall_scores: Any, clause_count: int) -> tuple[list[float], int]:
    """Return clause recall probabilities and the number of participants.

    Each participant is represented by a list of recalled clause indices.  A
    clause receives the fraction of participants whose list contains it.
    Invalid and out-of-range indices are ignored rather than silently being
    mapped to another clause.
    """
    if recall_scores is None:
        raise ValueError("Recall annotations are missing")

    participants = list(recall_scores)
    hits = [0] * clause_count
    all_indices = [
        int(index)
        for recalled in participants
        if recalled is not None
        for index in recalled
        if isinstance(index, (int, float)) or str(index).strip().isdigit()
    ]
    if all_indices and 0 in all_indices:
        index_base = 0
    elif all_indices and all(1 <= index <= clause_count for index in all_indices):
        index_base = 1
    else:
        raise ValueError(
            "Recall indices are neither consistently zero-based nor one-based"
        )

    for recalled in participants:
        seen: set[int] = set()
        if recalled is None:
            continue
        for index in recalled:
            try:
                index = int(index) - index_base
            except (TypeError, ValueError):
                continue
            if 0 <= index < clause_count:
                seen.add(index)
        for index in seen:
            hits[index] += 1

    denominator = len(participants)
    if denominator == 0:
        return [0.0] * clause_count, 0
    return [count / denominator for count in hits], denominator


def process_story(path: Path) -> dict[str, Any]:
    record = load_pickle(path)
    narrative = str(get_field(record, "narrative")).strip()
    clauses = clean_clauses(get_field(record, "segmentation"))
    recalls = get_field(record, "recall scores", "recall_scores")
    recall_probability, participant_count = normalize_recall_scores(
        recalls, len(clauses)
    )

    if not narrative:
        raise ValueError("Narrative is empty")
    if not clauses:
        raise ValueError(f"No clauses found in {path}")
    if len(recall_probability) != len(clauses):
        raise ValueError("Recall annotation length does not match clause count")

    story_id = path.stem
    return {
        "story_id": story_id,
        "source_file": path.name,
        "condition": "scrambled" if "scrambled" in story_id else "intact",
        "narrative": narrative,
        "participant_count": participant_count,
        "clauses": [
            {
                "clause_id": index,
                "text": clause,
                "human_recall_probability": recall_probability[index],
            }
            for index, clause in enumerate(clauses)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-scrambled",
        action="store_true",
        help="Only include intact narrative conditions.",
    )
    args = parser.parse_args()

    files = sorted(args.source.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl story files found in {args.source}")

    stories = []
    errors = []
    for path in tqdm(files, desc="Preparing evaluation stories", unit="story"):
        if args.exclude_scrambled and "scrambled" in path.stem:
            continue
        try:
            stories.append(process_story(path))
        except Exception as exc:  # report all bad files, then fail clearly
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise RuntimeError("Could not process dataset files:\n" + "\n".join(errors))
    if not stories:
        raise ValueError("No stories remain after filtering.")

    output = {
        "schema_version": 2,
        "target": "human_recall_probability",
        "source": "llm-narrative-analysis/data/compiled_data",
        "story_count": len(stories),
        "stories": stories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    clause_count = sum(len(story["clauses"]) for story in stories)
    print(f"Wrote {len(stories)} stories and {clause_count} clauses to {args.output}")


if __name__ == "__main__":
    main()
