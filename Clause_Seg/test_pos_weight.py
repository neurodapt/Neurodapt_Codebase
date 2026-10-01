import csv
import random
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from dataset import OptimizedClauseBoundaryDataset, optimized_padding
from losses import weighted_boundary_bce_loss
from metrics import calculate_metrics, update_counts
from model import ClauseSegmentationModel
from tools import debug_print


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "train_optimized.pt"
VAL_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "validation_optimized.pt"
RESULTS_PATH = PROJECT_DIR / "pos_weight_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
POS_WEIGHTS = [1, 2, 5, 10]


def pos_weight_debug_print(pos_weight: int, message: str) -> None:
    debug_print(f"[pos_weight={pos_weight}] {message}")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(
    model: ClauseSegmentationModel,
    data_loader: DataLoader,
    pos_weight: int,
):
    model.eval()
    total_loss = 0.0
    val_tp = val_fp = val_fn = val_correct = val_total = 0

    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch["embeddings"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)
            padding_mask = boundaries == -100
            valid_mask = ~padding_mask
            logits = model(embeddings, padding_mask=padding_mask)
            loss = weighted_boundary_bce_loss(
                logits,
                boundaries,
                valid_mask,
                pos_weight=pos_weight,
            )
            total_loss += loss.item()

            tp, fp, fn, correct, total = update_counts(
                logits, boundaries, valid_mask
            )
            val_tp += tp
            val_fp += fp
            val_fn += fn
            val_correct += correct
            val_total += total

    metrics = calculate_metrics(
        val_tp, val_fp, val_fn, val_correct, val_total
    )
    return metrics, total_loss / len(data_loader)


def train_for_pos_weight(
    pos_weight: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    set_seed()
    pos_weight_debug_print(pos_weight, "Importing Libraries")
    pos_weight_debug_print(pos_weight, f"Seed: {SEED}")
    if torch.cuda.is_available():
        pos_weight_debug_print(
            pos_weight,
            f"Using GPU: {torch.cuda.get_device_name(DEVICE)}",
        )
    else:
        pos_weight_debug_print(pos_weight, f"Using device: {DEVICE}")

    pos_weight_debug_print(
        pos_weight,
        f"Loaded training samples:{len(train_loader.dataset)} Successfully!",
    )
    pos_weight_debug_print(
        pos_weight,
        f"Loaded validation samples:{len(val_loader.dataset)} Successfully!",
    )

    model = ClauseSegmentationModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_f1 = 0.0
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in tqdm.tqdm(
            train_loader,
            desc=f"[pos_weight={pos_weight}] Epoch {epoch + 1} training",
            leave=False,
        ):
            embeddings = batch["embeddings"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)
            padding_mask = boundaries == -100
            valid_mask = ~padding_mask

            optimizer.zero_grad()

            logits = model(embeddings, padding_mask=padding_mask)

            loss = weighted_boundary_bce_loss(
                logits,
                boundaries,
                valid_mask,
                pos_weight=pos_weight,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        train_metrics, _ = evaluate(model, train_loader, pos_weight)
        val_metrics, val_loss = evaluate(model, val_loader, pos_weight)
        scheduler.step(val_metrics["f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        pos_weight_debug_print(
            pos_weight,
            f"[Training] Epoch {epoch + 1}/{EPOCHS}  "
            f"loss={train_loss:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"F1={train_metrics['f1']:.4f}, "
            f"Precision={train_metrics['precision']:.4f}, "
            f"Recall={train_metrics['recall']:.4f}",
        )
        pos_weight_debug_print(
            pos_weight,
            f"[Validation] Epoch {epoch + 1}/{EPOCHS} "
            f"loss={val_loss:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"Precision={val_metrics['precision']:.4f}, "
            f"Recall={val_metrics['recall']:.4f}, "
            f"lr={current_lr:.2e}",
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
            epochs_without_improvement = 0

            pos_weight_debug_print(
                pos_weight,
                f"[Validation] New best model saved with F1={best_f1:.4f}",
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                pos_weight_debug_print(
                    pos_weight,
                    f"[Training] Early stopping after {epoch + 1} epochs; "
                    f"validation F1 did not improve for "
                    f"{EARLY_STOPPING_PATIENCE} epochs.",
                )
                break

    return best_epoch, best_metrics


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {VAL_PATH}")

    train_dataset = OptimizedClauseBoundaryDataset(TRAIN_PATH)
    val_dataset = OptimizedClauseBoundaryDataset(VAL_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=optimized_padding,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=optimized_padding,
    )

    rows = []
    for pos_weight in POS_WEIGHTS:
        print("=" * 20 + f"Training (pos_weight={pos_weight})" + "=" * 20)
        pos_weight_debug_print(
            pos_weight,
            "All imports done; everything defined, now starting training!",
        )
        best_epoch, metrics = train_for_pos_weight(
            pos_weight, train_loader, val_loader
        )
        row = {
            "pos_weight": pos_weight,
            "best_epoch": best_epoch,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "F1": metrics["f1"],
        }
        rows.append(row)
        print(
            f"pos_weight={pos_weight}, best_epoch={best_epoch}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"accuracy={row['accuracy']:.4f}, F1={row['F1']:.4f}"
        )

    print("=" * 20 + "Training Complete" + "=" * 20)

    with RESULTS_PATH.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()