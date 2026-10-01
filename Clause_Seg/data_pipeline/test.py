from pathlib import Path
from pprint import pprint
import json



PROJECT_DIR = Path(__file__).resolve().parent
file_path = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_train.conllu"
print(f"Project Directory: {PROJECT_DIR}")

with file_path.open("r", encoding="utf-8") as handle:
    print(handle.read(5000))  # Read and print the first 1000 characters of the file