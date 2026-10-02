"""Train MemoryRanker on the precomputed train/validation story splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import MemoryRanker
from plot import save_loss_plot

PROJECT_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAMES = (
    "all_mpnet_base_v2",
    "minilm",
    "bert_base_uncased",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MemoryRankerDataset(Dataset):
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Run data_pipeline/optimized_data.py first."
            )
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        if not self.data:
            raise ValueError(f"Dataset is empty: {path}")
        if any(not story["clauses"] for story in self.data):
            raise ValueError(f"Dataset contains a story with no clauses: {path}")

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
    padded_embeddings = pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0.0,
    )
    padded_scores = pad_sequence(scores, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(item) for item in scores])
    positions = torch.arange(padded_scores.size(1)).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)

    return {
        "embeddings": padded_embeddings,
        "scores": padded_scores,
        "padding_mask": padding_mask,
    }


def loss_function(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine target calibration with within-story ranking loss."""
    valid_scores = ~padding_mask
    pointwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predicted_scores[valid_scores],
        target_scores[valid_scores],
    )

    pairwise_losses = []
    for predicted, target, mask in zip(
        predicted_scores,
        target_scores,
        padding_mask,
    ):
        predicted = predicted[~mask]
        target = target[~mask]
        if len(target) < 2:
            continue

        target_difference = target[:, None] - target[None, :]
        comparable = target_difference != 0
        if comparable.any():
            direction = target_difference[comparable].sign()
            predicted_difference = (
                predicted[:, None] - predicted[None, :]
            )[comparable]
            pair_loss = torch.relu(0.2 - direction * predicted_difference)
            pairwise_losses.append(pair_loss.mean())

    ranking_loss = (
        torch.stack(pairwise_losses).mean()
        if pairwise_losses
        else predicted_scores.new_tensor(0.0)
    )
    return 0.5 * pointwise_loss + 0.5 * ranking_loss


def run_epoch(
    model: MemoryRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    progress_label = "Training" if training else "Validation"
    for batch in tqdm(
        loader,
        desc=progress_label,
        leave=False,
        unit="batch",
    ):
        embeddings = batch["embeddings"].to(DEVICE)
        scores = batch["scores"].to(DEVICE)
        padding_mask = batch["padding_mask"].to(DEVICE)

        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            predicted_scores = model(embeddings, padding_mask=padding_mask)
            loss = loss_function(predicted_scores, scores, padding_mask)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(embedding_model_name: str) -> None:
    data_dir = PROJECT_DIR / "data_pipeline" / "optimized_data" / embedding_model_name
    train_path = data_dir / "train_optimized.pt"
    validation_path = data_dir / "validation_optimized.pt"
    output_dir = PROJECT_DIR / "models" / embedding_model_name
    model_path = output_dir / "memory_ranker_model_best.pt"
    loss_plot_path = output_dir / "training_loss.png"

    seed_everything()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Seed: {SEED}")
    print(f"Training {embedding_model_name} on device: {DEVICE}")

    train_dataset = MemoryRankerDataset(train_path)
    validation_dataset = MemoryRankerDataset(validation_path)
    if train_dataset.embedding_dim != validation_dataset.embedding_dim:
        raise ValueError("Train and validation embedding dimensions do not match")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=padding,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=padding,
    )
    model = MemoryRanker(input_dim=train_dataset.embedding_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        training_loss = run_epoch(model, train_loader, optimizer)
        validation_loss = run_epoch(model, validation_loader)
        scheduler.step(validation_loss)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training Loss: {training_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    save_loss_plot(training_losses, validation_losses, loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")


def main(model_name: str | None = None) -> None:
    model_names = (model_name,) if model_name else EMBEDDING_MODEL_NAMES
    for name in model_names:
        if name not in EMBEDDING_MODEL_NAMES:
            raise ValueError(f"Unknown embedding model: {name}")
        train_model(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MemoryRanker models.")
    parser.add_argument(
        "--model",
        choices=("all",) + EMBEDDING_MODEL_NAMES,
        default="all",
        help="Model to train, or all three models (default).",
    )
    args = parser.parse_args()
    main(None if args.model == "all" else args.model)
