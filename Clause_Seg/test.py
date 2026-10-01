from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OptimizedClauseBoundaryDataset, optimized_padding
from metrics import calculate_metrics, update_counts
from model import ClauseSegmentationModel


PROJECT_DIR = Path(__file__).resolve().parent
TEST_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "test_optimized.pt"
MODEL_PATH = PROJECT_DIR / "clause_segmentation_model_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4


def main() -> None:
	if not TEST_PATH.exists():
		raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

	test_dataset = OptimizedClauseBoundaryDataset(TEST_PATH)
	test_loader = DataLoader(
		test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		collate_fn=optimized_padding,
	)

	model = ClauseSegmentationModel().to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
	model.eval()

	tp = fp = fn = correct = total = 0
	with torch.no_grad():
		for batch in test_loader:
			embeddings = batch["embeddings"].to(DEVICE)
			boundaries = batch["boundaries"].to(DEVICE)
			valid_mask = boundaries != -100

			padding_mask = boundaries == -100
			valid_mask = ~padding_mask

			logits = model(embeddings, padding_mask=padding_mask)
			batch_tp, batch_fp, batch_fn, batch_correct, batch_total = update_counts(
				logits,
				boundaries,
				valid_mask,
				threshold=0.39,
			)
			tp += batch_tp
			fp += batch_fp
			fn += batch_fn
			correct += batch_correct
			total += batch_total

	results = calculate_metrics(tp, fp, fn, correct, total)
	print(f"Evaluated final model on {len(test_dataset)} test samples")
	print(f"Using device: {DEVICE}")
	for metric, value in results.items():
		print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
	main()