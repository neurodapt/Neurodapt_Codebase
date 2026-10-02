from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PAIRED_DATASET_PATH = HERE / "Breithaupt_Cleaned.csv"
MODEL_DIR = HERE / "models"
EMBEDDED_DATASET_DIR = HERE / "Embedded_dataset"
BERT_MODEL_DIR = PROJECT_ROOT / "Clause_Seg" / "models" / "bert-base-uncased"
BERT_MODEL_ID = "google-bert/bert-base-uncased"

MODEL_IDS = {
    "all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bert_base_uncased": BERT_MODEL_ID,
}
OUTPUT_PATHS = {
    name: EMBEDDED_DATASET_DIR / f"embedded_stories_{name}.pt"
    for name in MODEL_IDS
}


@lru_cache(maxsize=None)
def _load_cached_sentence_transformer(name: str, device: str) -> SentenceTransformer:
    """Download an encoder if needed, cache it locally, and keep it resident."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_id = MODEL_IDS[name]
    local_dir = MODEL_DIR / name
    if (local_dir / "modules.json").is_file():
        model = SentenceTransformer(str(local_dir), device=device)
    else:
        print(f"Downloading/loading {model_id}...")
        model = SentenceTransformer(
            model_id,
            device=device,
            cache_folder=str(MODEL_DIR / "huggingface_cache"),
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(local_dir))

    print(f"Ready: {model_id} ({local_dir}) on {device}")
    return model


def _bert_model_is_ready() -> bool:
    return (
        (BERT_MODEL_DIR / "config.json").is_file()
        and any(
            (BERT_MODEL_DIR / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        and any(
            (BERT_MODEL_DIR / filename).is_file()
            for filename in ("tokenizer.json", "vocab.txt")
        )
    )


@lru_cache(maxsize=None)
def _load_cached_bert(device: str) -> tuple[AutoTokenizer, AutoModel]:
    """Load the same BERT checkpoint used by the clause segmenter."""
    BERT_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = MODEL_DIR / "huggingface_cache"

    if _bert_model_is_ready():
        source = str(BERT_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModel.from_pretrained(source, local_files_only=True)
    else:
        print(f"Downloading/loading {BERT_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(
            BERT_MODEL_ID,
            cache_dir=str(cache_dir),
        )
        model = AutoModel.from_pretrained(
            BERT_MODEL_ID,
            cache_dir=str(cache_dir),
        )
        BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(BERT_MODEL_DIR)
        model.save_pretrained(BERT_MODEL_DIR, safe_serialization=True)

    selected_device = torch.device(device)
    model = model.to(selected_device)
    model.eval()
    print(f"Ready: {BERT_MODEL_ID} ({BERT_MODEL_DIR}) on {selected_device}")
    return tokenizer, model


class EmbeddingModels:
    """Load encoders on first use and keep them resident for this process."""

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.local_models: dict[str, SentenceTransformer] = {}

    def _load_sentence_transformer(self, name: str) -> SentenceTransformer:
        if name in self.local_models:
            return self.local_models[name]
        model = _load_cached_sentence_transformer(name, self.device)
        self.local_models[name] = model
        return model

    def load_model(self, model_name: str) -> SentenceTransformer:
        """Load and warm the named encoder before inference starts."""
        if model_name not in MODEL_IDS:
            raise KeyError(f"Unknown embedding model: {model_name}")
        if model_name == "bert_base_uncased":
            return _load_cached_bert(self.device)[1]
        return self._load_sentence_transformer(model_name)

    def encode(self, model_name: str, texts: list[str]) -> np.ndarray:
        if model_name not in MODEL_IDS:
            raise KeyError(f"Unknown embedding model: {model_name}")
        if not texts:
            dimension = self.dimension(model_name)
            return np.empty((0, dimension), dtype=np.float32)

        if model_name == "bert_base_uncased":
            tokenizer, model = _load_cached_bert(self.device)
            encoding = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoding = {
                key: value.to(self.device)
                for key, value in encoding.items()
            }
            with torch.no_grad():
                hidden_states = model(**encoding).last_hidden_state
            attention_mask = encoding["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden_states * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1).clamp_min(1.0)
            values = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        else:
            values = self._load_sentence_transformer(model_name).encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return values

    def dimension(self, model_name: str) -> int:
        if model_name == "bert_base_uncased":
            return int(_load_cached_bert(self.device)[1].config.hidden_size)
        return int(self._load_sentence_transformer(model_name).get_sentence_embedding_dimension())


@lru_cache(maxsize=None)
def _get_models(device: str) -> EmbeddingModels:
    return EmbeddingModels(device=device)


def embed(text: str, model_name: str = "minilm") -> np.ndarray:
    """Convenience single-text helper used by inference scripts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _get_models(device).encode(model_name, [text])[0]


def embed_all_stories() -> dict[str, Path]:
    """Create one embedded-story file per model, preserving the existing schema."""
    if not PAIRED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Source stories not found: {PAIRED_DATASET_PATH}")
    EMBEDDED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Construct the wrappers; each encoder is downloaded/loaded on first use.
    models = EmbeddingModels()

    import csv
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Clause_Segmentation import FullClauseSegmentation

    clause_model_path = PROJECT_ROOT / "Clause_Seg" / "clause_segmentation_model_best.pt"
    if not clause_model_path.exists():
        raise FileNotFoundError(f"Clause segmentation checkpoint not found: {clause_model_path}")
    device = torch.device(models.device)
    segmenter = FullClauseSegmentation(str(clause_model_path), device=device)

    with PAIRED_DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No source stories found in {PAIRED_DATASET_PATH}")

    embedded_by_model: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_IDS
    }
    for story_id, row in enumerate(tqdm(rows, desc="Segmenting and embedding stories")):
        original_text = (row.get("Original text") or "").strip()
        retold_texts = [
            (row.get(column) or "").strip()
            for column in ("G1 Story", "G2 Retelling", "G3 Retelling")
        ]
        original_clauses = segmenter.segment(original_text)
        retold_clauses = [segmenter.segment(text) for text in retold_texts]
        all_texts = original_clauses + [
            clause for group in retold_clauses for clause in group
        ]
        encoded = {name: models.encode(name, all_texts) for name in MODEL_IDS}

        for name in MODEL_IDS:
            vectors = encoded[name]
            original_count = len(original_clauses)
            cursor = original_count
            retold_vectors = []
            for clauses in retold_clauses:
                next_cursor = cursor + len(clauses)
                retold_vectors.append(
                    torch.from_numpy(vectors[cursor:next_cursor].copy())
                    if clauses
                    else torch.empty((0, models.dimension(name)), dtype=torch.float32)
                )
                cursor = next_cursor

            embedded_by_model[name].append(
                {
                    "story_id": story_id,
                    "original_clauses": original_clauses,
                    "retold_clauses": retold_clauses,
                    "original_embeddings": torch.from_numpy(
                        vectors[:original_count].copy()
                    ),
                    "retold_embeddings": retold_vectors,
                }
            )

    for name, stories in embedded_by_model.items():
        output_path = OUTPUT_PATHS[name]
        torch.save(stories, output_path)
        print(f"Saved {len(stories)} stories to {output_path}")
    return OUTPUT_PATHS


if __name__ == "__main__":
    embed_all_stories()