from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "optimized_data"
EMBEDDING_MODEL_NAMES = (
    "all_mpnet_base_v2",
    "minilm",
    "bert_base_uncased",
)
DEFAULT_EMBEDDING_MODEL = "all_mpnet_base_v2"

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


def _optimize_model(model_name: str) -> None:
    if model_name not in EMBEDDING_MODEL_NAMES:
        valid_models = ", ".join(EMBEDDING_MODEL_NAMES)
        raise ValueError(f"Unknown embedding model: {model_name}. Choose one of: {valid_models}")

    embedded_dataset_path = (
        HERE / "Embedded_dataset" / f"embedded_stories_{model_name}.pt"
    )
    model_output_dir = OUTPUT_DIR / model_name
    split_paths = {
        split: model_output_dir / f"{split}_optimized.pt"
        for split in ("train", "validation", "test")
    }

    if not embedded_dataset_path.exists():
        raise FileNotFoundError(
            f"Expected {model_name} embeddings at {embedded_dataset_path}. "
            "Run embedding.py first."
        )

    embedded_stories = torch.load(
        embedded_dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    scored_stories = [
        _score_story(story)
        for story in tqdm(
            embedded_stories,
            desc="Scoring stories",
            unit="story",
        )
    ]

    splits = _split_stories(scored_stories)

    model_output_dir.mkdir(parents=True, exist_ok=True)

    for name, stories in splits.items():
        output_path = split_paths[name]
        torch.save(stories, output_path)
        print(f"Saved {len(stories)} {name} stories to {output_path}")


def main(model_name: str | None = None) -> None:
    model_names = (model_name,) if model_name else EMBEDDING_MODEL_NAMES
    for name in model_names:
        _optimize_model(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test datasets for an embedding model."
    )
    parser.add_argument(
        "--model",
        choices=("all",) + EMBEDDING_MODEL_NAMES,
        default="all",
        help="Embedding model to optimize, or all three models (default).",
    )
    args = parser.parse_args()
    main(None if args.model == "all" else args.model)