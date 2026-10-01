from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader

from dataset import OptimizedClauseBoundaryDataset, optimized_padding
from metrics import calculate_metrics
from model import ClauseSegmentationModel


PROJECT_DIR = Path(__file__).resolve().parent
VAL_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "validation_optimized.pt"
MODEL_PATH = PROJECT_DIR / "clause_segmentation_model_best.pt"
RESULTS_PATH = PROJECT_DIR / "threshold_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
THRESHOLDS = [threshold / 100 for threshold in range(10, 91)]


def collect_predictions(val_loader: DataLoader, model: ClauseSegmentationModel):
    probabilities = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch["embeddings"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)
            padding_mask = boundaries == -100
            valid_mask = ~padding_mask

            logits = model(embeddings, padding_mask=padding_mask)
            probabilities.append(torch.sigmoid(logits)[valid_mask].cpu())
            targets.append(boundaries[valid_mask].cpu())

    return torch.cat(probabilities), torch.cat(targets)


def main() -> None:
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation dataset not found: {VAL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    val_dataset = OptimizedClauseBoundaryDataset(VAL_PATH)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=optimized_padding,
    )

    model = ClauseSegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    probabilities, targets = collect_predictions(val_loader, model)

    print(f"Evaluated {len(val_dataset)} validation samples")
    print(f"Using device: {DEVICE}")
    print("threshold  precision  recall  accuracy  f1")

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities > threshold).long()
        tp = int(((predictions == 1) & (targets == 1)).sum())
        fp = int(((predictions == 1) & (targets == 0)).sum())
        fn = int(((predictions == 0) & (targets == 1)).sum())
        correct = int((predictions == targets).sum())
        total = targets.numel()
        results = calculate_metrics(tp, fp, fn, correct, total)
        row = {
            "threshold": threshold,
            "precision": results["precision"],
            "recall": results["recall"],
            "accuracy": results["accuracy"],
            "f1": results["f1"],
        }
        rows.append(row)

        print(
            f"{threshold:9.2f}  "
            f"{results['precision']:.4f}    "
            f"{results['recall']:.4f}  "
            f"{results['accuracy']:.4f}  "
            f"{results['f1']:.4f}"
        )

    with RESULTS_PATH.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["f1"])
    print(
        f"Best threshold: {best['threshold']:.2f} | "
        f"precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, "
        f"accuracy={best['accuracy']:.4f}, "
        f"F1={best['f1']:.4f}"
    )
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
