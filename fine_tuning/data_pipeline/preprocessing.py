# Convert every Parquet file in raw_data into a filtered CSV.
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


data_dir = Path(__file__).resolve().parent
input_dir = data_dir / "raw_data"
output_dir = data_dir / "processed_data"
output_dir.mkdir(parents=True, exist_ok=True)

parquet_paths = sorted(input_dir.glob("*.parquet"))
if not parquet_paths:
	raise FileNotFoundError(f"No Parquet files found in {input_dir}")

for input_path in parquet_paths:
	output_path = output_dir / f"{input_path.stem}.csv"
	parquet_file = pq.ParquetFile(input_path)
	first_batch = True

	with output_path.open("w", newline="", encoding="utf-8") as csv_file:
		with tqdm(
			total=parquet_file.metadata.num_rows,
			unit="rows",
			desc=f"Converting {input_path.name}",
		) as progress:
			for batch in parquet_file.iter_batches(batch_size=1000):
				batch.to_pandas().to_csv(
					csv_file,
					index=False,
					header=first_batch,
				)
				first_batch = False
				progress.update(batch.num_rows)

	# Keep only the serial number and text for rows with at least 50 words.
	df = pd.read_csv(output_path)
	original_count = len(df)
	word_counts = df["text"].fillna("").astype(str).str.split().str.len()
	df = df[word_counts >= 50].copy()
	df.insert(0, "serial_number", range(1, len(df) + 1))
	df = df[["serial_number", "text"]]
	df.to_csv(output_path, index=False)

	print(f"\nWrote {output_path}")
	print(f"Removed {original_count - len(df)} datapoints with fewer than 50 words")
	print(f"Remaining datapoints: {len(df)}")

	print("First 5 rows:")
	for _, row in df.head(5).iterrows():
		text = str(row["text"])
		readable_text = text.replace("\\n", "\n")
		preview = readable_text[:500]
		if len(readable_text) > 500:
			preview += "..."
		print(f"\nserial_number: {row['serial_number']}")
		print(f"character_count: {len(text)}")
		print("text_preview:")
		print(preview)
