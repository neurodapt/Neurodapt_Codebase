from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
EMBEDDED_DATASET_PATH = HERE / "Embedded_dataset" / "embedded_stories_all_mpnet_base_v2.pt"
OUTPUT_DIR = HERE / "optimized_data"

SPLIT_PATHS = {
    "train": OUTPUT_DIR / "train_optimized.pt",
    "validation": OUTPUT_DIR / "validation_optimized.pt",
    "test": OUTPUT_DIR / "test_optimized.pt",
}

SEED = 42
random.seed(SEED)


def _normalize_within_story(values: torch.Tensor) -> torch.Tensor:
    value_min = values.min()
    value_range = values.max() - value_min

    if value_range > 0:
        return (values - value_min) / value_range

    return torch.full_like(values, 0.5)


def _retelling_scores(original_embedding: torch.Tensor,retold_embeddings: list[torch.Tensor]) -> torch.Tensor:
    best_matches = []

    for retelling in retold_embeddings:
        if retelling.numel() == 0:
            best_matches.append(torch.tensor(0.0, dtype=torch.float32))
        else:
            best_match = (retelling @ original_embedding).max()
            best_matches.append(best_match)

    return torch.stack(best_matches).sum()


def _score_story(story: dict[str, Any]) -> dict[str, Any]:
    original_embeddings = story["original_embeddings"].float()

    # 1. Calculate RAW summed retelling score for every clause
    retelling_scores = []

    for index in range(len(story["original_clauses"])):
        retelling_score = _retelling_scores(original_embeddings[index],story["retold_embeddings"],)

        retelling_scores.append(retelling_score)

    retelling = torch.stack(retelling_scores)

    # 2. Normalize summed retelling scores independently within the story
    normalized_retelling = _normalize_within_story(retelling)

    # 3. Use retelling scores as the targets
    combined_targets = normalized_retelling

    # 4. Final normalization within the story
    final_targets = _normalize_within_story(combined_targets)

    # 5. Build final clause records
    clauses = []

    for index, clause_text in enumerate(story["original_clauses"]):
        clauses.append(
            {
                "text": clause_text,
                "embedding": original_embeddings[index],
                "retelling_score": float(normalized_retelling[index]),
                "target": float(final_targets[index]),
            }
        )

    return {"story_id": story["story_id"],"clauses": clauses,}


def _split_stories(stories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(stories)

    random.seed(SEED)
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.10)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def main() -> None:
    if not EMBEDDED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Expected MPNet embeddings at {EMBEDDED_DATASET_PATH}. Run embedding.py first.")

    embedded_stories = torch.load(EMBEDDED_DATASET_PATH,map_location="cpu",weights_only=False)

    scored_stories = [
        _score_story(story)
        for story in tqdm(
            embedded_stories,
            desc="Scoring stories",
            unit="story",
        )
    ]

    splits = _split_stories(scored_stories)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, stories in splits.items():
        output_path = SPLIT_PATHS[name]
        torch.save(stories, output_path)
        print(f"Saved {len(stories)} {name} stories to {output_path}")


if __name__ == "__main__":
    main()