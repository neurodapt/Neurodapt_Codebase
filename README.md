# Major Project Research

Research code for clause segmentation, language-model fine-tuning, and narrative-memory ranking. This repository is an incomplete working snapshot rather than a fully reproducible release.

## Components

- `Clause_Seg/` contains the clause-segmentation model, training code, tests, and the AMR-based dataset files.
- `fine_tuning/` contains experimental fine-tuning, evaluation, and profiling scripts.
- `mem_ranker/` contains the MemoryRanker model, training code, and narrative-memory evaluation pipeline.
- `main.tex` contains the project report source.

## Setup

Python 3.11 or 3.12 is required. The project dependencies are defined in `pyproject.toml` and `uv.lock`.

```powershell
uv sync
```

A CUDA-capable PyTorch installation may be required for training. Check the PyTorch and CUDA versions in `pyproject.toml` before changing the environment.

## Clause Segmentation

Run commands from the repository root, for example:

```powershell
python Clause_Seg/test.py
python Clause_Seg/train.py
```

The trained checkpoint is stored at `Clause_Seg/clause_segmentation_model_best.pt`.

## MemoryRanker Evaluation

The narrative evaluation data is stored in `mem_ranker/eval_pipeline/llm-narrative-analysis/data/compiled_data/`.

Regenerate the normalized evaluation JSON with:

```powershell
python mem_ranker/eval_pipeline/pre-processing.py
```

Run evaluation after providing the missing local embedding model described below:

```powershell
python mem_ranker/eval_pipeline/eval.py
```

## Incomplete or Missing Parts

- The fine-tuning experiments are incomplete; no end-to-end training run or final comparison is included.
- Fine-tuning datasets, downloaded base models, generated checkpoints, and output directories are intentionally not included in this repository.
- MemoryRanker evaluation expects the local SentenceTransformer model `mem_ranker/data_pipeline/models/all_mpnet_base_v2`, which is not included.
- Some intermediate MemoryRanker and fine-tuning data artifacts are ignored by `.gitignore` and must be regenerated or supplied separately.
- The available model checkpoints are research artifacts and are not accompanied by a final model card, published metrics, or reproducibility report.
- The original narrative-analysis repository's notebooks, figures, auxiliary datasets, and nested Git metadata were removed. Only the compiled story pickles required by the preprocessing pipeline are retained.
- Training and evaluation results may depend on local GPU availability, package versions, random seeds, and external model downloads.

## Status

This codebase is suitable for continuing experiments and inspecting the current pipelines. It should not yet be treated as a complete reproduction package.
