"""Plot training and validation losses for MemoryRanker."""

from pathlib import Path

try:
	import matplotlib.pyplot as plt
except ModuleNotFoundError:
	plt = None


LOSS_PLOT_PATH = Path(__file__).resolve().parent / "training_loss.png"


def save_loss_plot(training_losses: list[float], validation_losses: list[float]) -> None:
	if plt is None:
		print("matplotlib is not installed; skipping loss plot")
		return

	figure, axis = plt.subplots(figsize=(10, 6))
	epochs = range(1, len(training_losses) + 1)
	axis.plot(epochs, training_losses, marker="o", label="Training Loss")
	axis.plot(epochs, validation_losses, marker="o", label="Validation Loss")
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Loss")
	axis.set_title("Training vs Validation Loss")
	axis.grid(True, alpha=0.3)
	axis.legend()
	figure.tight_layout()
	figure.savefig(LOSS_PLOT_PATH, dpi=150)
	plt.close(figure)


def create_live_loss_plot() -> tuple[object, object, object, object] | None:
	if plt is None:
		print("matplotlib is not installed; skipping live loss plot")
		return None

	plt.ion()
	figure, axis = plt.subplots(figsize=(10, 6))
	training_line, = axis.plot([], [], marker="o", label="Training Loss")
	validation_line, = axis.plot([], [], marker="o", label="Validation Loss")
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Loss")
	axis.set_title("Training vs Validation Loss")
	axis.grid(True, alpha=0.3)
	axis.legend()
	figure.tight_layout()
	figure.show()
	return figure, axis, training_line, validation_line


def update_live_loss_plot(
	plot: tuple[object, object, object, object] | None,
	training_losses: list[float],
	validation_losses: list[float],
) -> None:
	if plot is None:
		return

	figure, axis, training_line, validation_line = plot
	epochs = list(range(1, len(training_losses) + 1))
	training_line.set_data(epochs, training_losses)
	validation_line.set_data(epochs, validation_losses)
	axis.relim()
	axis.autoscale_view()
	figure.canvas.draw()
	figure.canvas.flush_events()
	plt.pause(0.001)
