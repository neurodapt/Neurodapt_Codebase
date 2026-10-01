"""Evaluate the trained MemoryRanker on the held-out test split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_pipeline" / "optimized_data"
TEST_PATH = DATA_DIR / "test_optimized.pt"
MODEL_PATH = PROJECT_DIR / "memory_ranker_model_best.pt"
EXPECTED_EMBEDDING_DIMENSION = 768
SPLIT_NAMES = ("train", "validation", "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_pipeline.optimized_data import _score_story, _split_stories
from model import MemoryRanker


BATCH_SIZE = 8
EMBEDDING_MODEL_PATH = PROJECT_DIR / "data_pipeline" / "models" / "all_mpnet_base_v2"

MANUAL_STORIES = [
    {
        "story_id": "the-lighthouse",
        "clauses": [
            "Mara kept the old lighthouse running after the island lost power.",
            "At midnight, she saw a fishing boat drifting toward the rocks.",
            "She climbed the tower and lit the emergency beacon by hand.",
            "The boat changed course and its crew reached the harbor safely.",
        ],
    },
    {
        "story_id": "the-missing-key",
        "clauses": [
            "Jonas searched every drawer for the brass key to his grandmother's house.",
            "He found a muddy footprint beneath the kitchen window.",
            "A magpie dropped the key beside the garden gate.",
            "Jonas unlocked the house and discovered a letter on the table.",
        ],
    },
    {
        "story_id": "the-last-train",
        "clauses": [
            "The station clock stopped just before the last train arrived.",
            "Priya noticed that one passenger was still asleep on the platform bench.",
            "She woke him, and he handed her a red envelope without saying a word.",
            "When the train departed, Priya realized the envelope was addressed to her.",
        ],
    },
    {
        "story_id": "the-winter-garden",
        "clauses": [
            "In January, the school garden was buried under a layer of ice.",
            "The students built small shelters from fallen branches and glass jars.",
            "One green shoot appeared beneath the largest shelter.",
            "By spring, the class had grown enough vegetables to fill the cafeteria.",
        ],
    },
]


class MemoryRankerDataset(Dataset):
    def __init__(self, path: Path):
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        if not self.data:
            raise ValueError(f"Dataset is empty: {path}")
        self.embedding_dim = self.data[0]["clauses"][0]["embedding"].numel()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clauses = self.data[index]["clauses"]
        embeddings = torch.stack([clause["embedding"] for clause in clauses])
        targets = torch.tensor(
            [clause["target"] for clause in clauses],
            dtype=torch.float32,
        )
        return embeddings.float(), targets


def padding(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    embeddings, scores = zip(*batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    padded_scores = pad_sequence(scores, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(item) for item in scores])
    positions = torch.arange(padded_scores.size(1)).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    return {
        "embeddings": padded_embeddings,
        "scores": padded_scores,
        "padding_mask": padding_mask,
    }


def _assert_close(actual: float, expected: float, message: str) -> None:
    if abs(actual - expected) > 1e-6:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def test_scoring_math() -> None:
    original = torch.eye(2)

    matching_retelling = [
        torch.tensor([[1.0, 0.0]])
    ] * 3

    story = {
        "story_id": 0,
        "original_clauses": ["first", "second"],
        "original_embeddings": original,
        "retold_embeddings": matching_retelling,
    }

    scored = _score_story(story)["clauses"]

    _assert_close(
        scored[0]["retelling_score"],
        1.0,
        "retelling score",
    )

    _assert_close(
        scored[0]["target"],
        1.0,
        "normalized target",
    )

    _assert_close(
        scored[1]["retelling_score"],
        0.0,
        "unmatched retelling score",
    )

    _assert_close(
        scored[1]["target"],
        0.0,
        "second normalized target",
    )


def test_split_math() -> None:
    stories = [{"story_id": index} for index in range(10)]
    splits = _split_stories(stories)
    lengths = [len(splits[name]) for name in SPLIT_NAMES]
    if lengths != [7, 1, 2]:
        raise AssertionError(f"Expected 70:10:20 split for 10 stories, got {lengths}")

    ids = [story["story_id"] for stories in splits.values() for story in stories]
    if len(ids) != len(set(ids)) or set(ids) != set(range(10)):
        raise AssertionError("Story IDs overlap or are missing across splits")


def test_saved_splits() -> None:
    split_data = {}
    for name in SPLIT_NAMES:
        path = DATA_DIR / f"{name}_optimized.pt"
        if not path.exists():
            raise AssertionError(
                f"Missing {path}. Run data_pipeline/optimized_data.py first."
            )
        split_data[name] = torch.load(path, map_location="cpu", weights_only=False)

    story_ids = []
    for name, stories in split_data.items():
        if not stories:
            raise AssertionError(f"{name} split is empty")
        for story in stories:
            story_ids.append(story["story_id"])
            if not story["clauses"]:
                raise AssertionError(f"Story {story['story_id']} has no clauses")
            for clause in story["clauses"]:
                if clause["embedding"].numel() != EXPECTED_EMBEDDING_DIMENSION:
                    raise AssertionError(
                        "A clause does not contain a 768-dimensional MPNet embedding"
                    )
                if not 0.0 <= clause["target"] <= 1.0:
                    raise AssertionError("A clause target is outside [0, 1]")

    if len(story_ids) != len(set(story_ids)):
        raise AssertionError("Story IDs overlap across saved splits")


def evaluate_model() -> dict[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. Run train.py first."
        )

    dataset = MemoryRankerDataset(TEST_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=padding,
    )
    model = MemoryRanker(input_dim=dataset.embedding_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing model", unit="batch"):
            embeddings = batch["embeddings"].to(DEVICE)
            scores = batch["scores"].to(DEVICE)
            padding_mask = batch["padding_mask"].to(DEVICE)
            logits = model(embeddings, padding_mask=padding_mask)
            valid = ~padding_mask
            predictions.append(torch.sigmoid(logits[valid]).cpu())
            targets.append(scores[valid].cpu())

    predicted = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    absolute_error = np.abs(predicted - target)
    squared_error = (predicted - target) ** 2
    comparable = 0
    correct_order = 0
    for story in dataset.data:
        story_embeddings = torch.stack(
            [clause["embedding"] for clause in story["clauses"]]
        ).unsqueeze(0).to(DEVICE)
        story_targets = np.asarray(
            [clause["target"] for clause in story["clauses"]],
            dtype=np.float32,
        )
        with torch.no_grad():
            story_predictions = torch.sigmoid(model(story_embeddings))[0].cpu().numpy()
        for left in range(len(story_targets)):
            for right in range(left + 1, len(story_targets)):
                target_difference = story_targets[left] - story_targets[right]
                if target_difference == 0:
                    continue
                comparable += 1
                prediction_difference = story_predictions[left] - story_predictions[right]
                if target_difference * prediction_difference > 0:
                    correct_order += 1

    metrics = {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "pairwise_accuracy": (
            float(correct_order / comparable) if comparable else 0.0
        ),
    }
    return metrics


def manual_story_demo() -> None:
    """Print model scores for a few hand-written stories."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}. Run train.py first."
        )
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {EMBEDDING_MODEL_PATH}. "
            "Run data_pipeline/embedding.py first."
        )

    encoder = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device=str(DEVICE))
    model = MemoryRanker(input_dim=EXPECTED_EMBEDDING_DIMENSION).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    print("\nManual story behavior:")
    for story in MANUAL_STORIES:
        embeddings = torch.from_numpy(
            encoder.encode(
                story["clauses"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        ).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            scores = torch.sigmoid(model(embeddings))[0].cpu().tolist()

        ranked = sorted(
            zip(scores, story["clauses"]),
            key=lambda item: item[0],
            reverse=True,
        )
        print(f"\n{story['story_id']} (highest score first)")
        for rank, (score, clause) in enumerate(ranked, start=1):
            print(f"  {rank}. {score:.4f} | {clause}")


def main() -> None:
    test_scoring_math()
    test_split_math()
    test_saved_splits()
    metrics = evaluate_model()
    print("Test results:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    manual_story_demo()


if __name__ == "__main__":
    main()
