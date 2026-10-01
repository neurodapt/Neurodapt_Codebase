"""Evaluate MemoryRanker against human clause-recall probabilities.

Run from the mem_ranker directory:

    python eval_pipeline/eval.py

The script evaluates each story independently because MemoryRanker uses
clause-level self-attention and therefore produces context-dependent scores.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
DEFAULT_DATA = EVAL_DIR / "narrative_memory_eval.json"
DEFAULT_MODEL = PROJECT_DIR / "memory_ranker_model_best.pt"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model import MemoryRanker  # noqa: E402


def rankdata(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(predicted: np.ndarray, target: np.ndarray) -> float:
    if len(predicted) < 2:
        return 0.0
    a, b = rankdata(predicted), rankdata(target)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau(predicted: np.ndarray, target: np.ndarray) -> float:
    concordant = discordant = comparable = 0
    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            target_delta = target[i] - target[j]
            if target_delta == 0:
                continue
            comparable += 1
            predicted_delta = predicted[i] - predicted[j]
            if predicted_delta == 0:
                continue
            if target_delta * predicted_delta > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / comparable if comparable else 0.0


def pairwise_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    correct = total = 0
    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            target_delta = target[i] - target[j]
            if target_delta == 0:
                continue
            total += 1
            predicted_delta = predicted[i] - predicted[j]
            if target_delta * predicted_delta > 0:
                correct += 1
    return correct / total if total else 0.0


def ndcg(predicted: np.ndarray, target: np.ndarray, k: int) -> float:
    k = min(k, len(target))
    if k == 0:
        return 0.0
    predicted_order = np.argsort(-predicted)[:k]
    ideal_order = np.argsort(-target)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2.0 ** target[predicted_order] - 1.0) * discounts)
    ideal = np.sum((2.0 ** target[ideal_order] - 1.0) * discounts)
    return float(dcg / ideal) if ideal > 0 else 0.0


def metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = predicted - target
    predicted_top = set(np.argsort(-predicted)[: min(5, len(target))])
    target_top = set(np.argsort(-target)[: min(5, len(target))])
    top_count = min(5, len(target))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "spearman": spearman(predicted, target),
        "kendall_tau": kendall_tau(predicted, target),
        "pairwise_accuracy": pairwise_accuracy(predicted, target),
        "ndcg_at_1": ndcg(predicted, target, 1),
        "ndcg_at_3": ndcg(predicted, target, 3),
        "ndcg_at_5": ndcg(predicted, target, 5),
        "top5_overlap": len(predicted_top & target_top) / top_count,
    }


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize scores within one story to [0, 1]."""
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    maximum = np.max(values)
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def metric_std(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.std([row[key] for row in rows])) for key in keys}


def metric_median(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.median([row[key] for row in rows])) for key in keys}


def evaluate(args: argparse.Namespace) -> dict:
    data = json.loads(args.data.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    encoder = SentenceTransformer(str(args.embedding_model), device=str(device))
    embedding_dim = encoder.get_sentence_embedding_dimension()
    model = MemoryRanker(input_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    model_rows = []
    random_rows = []
    position_rows = []
    embedding_rows = []
    model_beats_baseline = {
        "random": 0,
        "position": 0,
        "embedding_similarity": 0,
    }
    story_results = []
    rng = random.Random(args.seed)

    for story in tqdm(data["stories"], desc="Evaluating stories", unit="story"):
        clauses = story["clauses"]
        texts = [clause["text"] for clause in clauses]
        target = np.asarray(
            [clause["human_recall_probability"] for clause in clauses],
            dtype=float,
        )
        embeddings = torch.from_numpy(
            np.asarray(
                encoder.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        ).unsqueeze(0).to(device)

        # The training pipeline uses normalized MPNet embeddings, so this is
        # a comparable embedding-only baseline using the same representation.
        clause_embeddings = embeddings.squeeze(0).cpu().numpy()
        story_embedding = np.mean(clause_embeddings, axis=0)
        story_embedding /= max(np.linalg.norm(story_embedding), 1e-12)
        embedding_scores = clause_embeddings @ story_embedding

        with torch.no_grad():
            predicted = torch.sigmoid(model(embeddings)).squeeze(0).cpu().numpy()

        random_scores = np.asarray([rng.random() for _ in clauses])
        # Normalize non-probability baselines per story for comparable errors.
        embedding_scores = minmax_normalize(embedding_scores)
        position_scores = minmax_normalize(np.arange(len(clauses), dtype=float))
        model_metric = metrics(predicted, target)
        random_metric = metrics(random_scores, target)
        position_metric = metrics(position_scores, target)
        embedding_metric = metrics(embedding_scores, target)

        model_rows.append(model_metric)
        random_rows.append(random_metric)
        position_rows.append(position_metric)
        embedding_rows.append(embedding_metric)
        for name, baseline_metric in (
            ("random", random_metric),
            ("position", position_metric),
            ("embedding_similarity", embedding_metric),
        ):
            if model_metric["pairwise_accuracy"] > baseline_metric["pairwise_accuracy"]:
                model_beats_baseline[name] += 1
        story_results.append({
            "story_id": story["story_id"],
            "condition": story["condition"],
            "participant_count": story["participant_count"],
            "metrics": model_metric,
            "predictions": [
                {"clause_id": index, "text": text, "prediction": float(score),
                 "human_recall_probability": float(target[index])}
                for index, (text, score) in enumerate(zip(texts, predicted))
            ],
        })

    result = {
        "model": str(args.model),
        "dataset": str(args.data),
        "device": str(device),
        "story_count": len(story_results),
        "model_metrics": mean_metrics(model_rows),
        "model_metric_std": metric_std(model_rows),
        "model_metric_median": metric_median(model_rows),
        "model_beats_baseline_by_story": model_beats_baseline,
        "random_baseline": mean_metrics(random_rows),
        "position_baseline": mean_metrics(position_rows),
        "embedding_similarity_baseline": mean_metrics(embedding_rows),
        "stories": story_results,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=PROJECT_DIR / "data_pipeline" / "models" / "all_mpnet_base_v2",
        help="Local SentenceTransformer used by the training pipeline.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Evaluation data not found: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.embedding_model.exists():
        raise FileNotFoundError(f"Embedding model not found: {args.embedding_model}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    result = evaluate(args)
    print(f"Evaluated {result['story_count']} stories on {result['device']}")
    for name in (
        "model_metrics",
        "random_baseline",
        "position_baseline",
        "embedding_similarity_baseline",
    ):
        print(f"\n{name}")
        for metric, value in result[name].items():
            print(f"  {metric}: {value:.4f}")
    print("\nmodel_metric_std")
    for metric, value in result["model_metric_std"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nmodel_metric_median")
    for metric, value in result["model_metric_median"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nmodel stories beating baseline by pairwise accuracy")
    for name, count in result["model_beats_baseline_by_story"].items():
        print(f"  {name}: {count}/{result['story_count']}")
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
