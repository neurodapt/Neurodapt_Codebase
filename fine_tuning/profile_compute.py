"""Profile GRPO setup, optimizer steps, and reward computation.

Run from this directory with the project environment active:
    python profile_compute.py --steps 3

Requires the same training dependencies and input artifacts as train.py. Runs
one warm-up optimizer step followed by the requested measured steps.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import TrainerCallback

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mem_ranker.model import MemoryRanker


def timed_call(bucket: dict[str, list[float]], name: str, function: Callable[..., Any]):
    """Wrap a callable and accumulate inclusive wall-clock durations."""
    def wrapped(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            bucket[name].append(time.perf_counter() - start)

    return wrapped


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepTimingCallback(TrainerCallback):
    """Measure optimizer steps and advance the PyTorch profiler schedule."""

    def __init__(self, profiler: Any, step_times: list[float]) -> None:
        self.profiler = profiler
        self.step_times = step_times
        self.step_start: float | None = None

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        del args, state, kwargs
        sync_cuda()
        self.step_start = time.perf_counter()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        del args, state, kwargs
        sync_cuda()
        if self.step_start is not None:
            self.step_times.append(time.perf_counter() - self.step_start)
            self.step_start = None
        self.profiler.step()
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3,
                        help="Measured steps after one warm-up step (default: 3).")
    parser.add_argument("--max-examples", type=int, default=64,
                        help="Maximum dataset rows to use (default: 64).")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "outputs" / "profile-run",
                        help="Directory for profiler trace and reports.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Device for reward models (default: auto).")
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_memory_ranker(checkpoint_path: str, device: str) -> MemoryRanker:
    ranker = MemoryRanker(input_dim=768).to(device)
    ranker.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    ranker.eval()
    return ranker


class ProfileRewardScorer:
    """MPNet + the project's trained ranker, for profiling this workspace."""

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        self.device = torch.device(device)
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=str(self.device),
        )
        self.ranker = load_memory_ranker(str(checkpoint_path), str(self.device))
        self.original_embeddings: dict[str, np.ndarray] = {}

    def cache_originals(self, originals: list[str]) -> None:
        unique = list(dict.fromkeys(text.strip() for text in originals if text.strip()))
        uncached = [text for text in unique if text not in self.original_embeddings]
        if not uncached:
            return
        vectors = self.encoder.encode(
            uncached,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.original_embeddings.update(
            (text, np.asarray(vector, dtype=np.float32))
            for text, vector in zip(uncached, vectors)
        )

    def score_pairs(self, originals: list[str], rewrites: list[str]) -> list[float]:
        if len(originals) != len(rewrites):
            raise ValueError("originals and rewrites must have equal lengths")
        if not originals:
            return []
        self.cache_originals(originals)
        vectors = self.encoder.encode(
            rewrites,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        paired = np.stack([
            np.stack((self.original_embeddings[original.strip()], vector))
            for original, vector in zip(originals, vectors)
        ])
        tensor = torch.as_tensor(paired, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.ranker(tensor))
            differences = probabilities[:, 1] - probabilities[:, 0]
        return differences.cpu().tolist()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", "")).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion).strip()


def make_reward_function(scorer: ProfileRewardScorer):
    """Build the same pairwise score and word-limit reward used by training."""
    def reward_function(
        prompts: list[Any],
        completions: list[Any],
        target_clause: list[str],
        **kwargs: Any,
    ) -> list[float]:
        del prompts, kwargs
        rewards = [-1.5] * len(completions)
        pending: list[tuple[int, str, str, int, int]] = []
        for index, (completion, original) in enumerate(zip(completions, target_clause)):
            original = str(original).strip()
            rewrite = completion_text(completion)
            original_count = len(original.split())
            rewrite_count = len(rewrite.split())
            if not original_count or not rewrite_count:
                rewards[index] = -2.5
            else:
                pending.append((index, original, rewrite, original_count, rewrite_count))

        if pending:
            scores = scorer.score_pairs(
                [item[1] for item in pending],
                [item[2] for item in pending],
            )
            for (index, _, _, original_count, rewrite_count), score in zip(pending, scores):
                if 2 * rewrite_count > 3 * original_count:
                    excess_ratio = rewrite_count / (1.5 * original_count) - 1.0
                    rewards[index] = -1.5 - excess_ratio + 0.25 * float(score)
                else:
                    rewards[index] = float(score)
        return rewards

    return reward_function


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.max_examples < 1:
        raise SystemExit("--steps and --max-examples must both be positive")

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "grpo_trace.json"
    report_path = args.output_dir / "profile_summary.json"
    text_path = args.output_dir / "profiler_operator_summary.txt"
    setup_seconds: dict[str, float] = {}
    reward_times: dict[str, list[float]] = defaultdict(list)
    step_times: list[float] = []

    def measure_setup(label: str, function: Callable[[], Any]):
        start = time.perf_counter()
        result = function()
        sync_cuda()
        setup_seconds[label] = time.perf_counter() - start
        print(f"Setup {label}: {setup_seconds[label]:.2f}s")
        return result

    model_path = HERE / "models" / "Qwen2.5-0.5B-Instruct"
    dataset_path = HERE / "data_pipeline" / "preprocessed_grpo.pt"
    ranker_path = PROJECT_ROOT / "mem_ranker" / "memory_ranker_model_best.pt"
    for path in (model_path, dataset_path, ranker_path):
        if not path.exists():
            raise FileNotFoundError(f"Required profiling input not found: {path}")

    tokenizer = measure_setup("policy_tokenizer_load", lambda: AutoTokenizer.from_pretrained(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_data = measure_setup(
        "dataset_load",
        lambda: torch.load(dataset_path, map_location="cpu", weights_only=False),
    )

    def prepare_examples() -> list[dict[str, Any]]:
        valid = [
            item for item in raw_data["examples"]
            if 0 <= item["target_index"] < len(item["clauses"])
        ]
        if len(valid) > args.max_examples:
            valid = random.Random(42).sample(valid, args.max_examples)

        rows = []
        for item in valid:
            clauses = item["clauses"]
            target_index = item["target_index"]
            target = clauses[target_index]["text"]
            word_count = len(target.split())
            context = [
                f"[TARGET] {clause['text']}" if i == target_index else clause["text"]
                for i, clause in enumerate(clauses)
            ]
            prompt = (
                "Rewrite ONLY the target clause.\n\n"
                "Keep its meaning unchanged. Do not add information.\n"
                "Do not use information from other clauses.\n"
                "Output exactly one sentence.\n"
                f"Keep the rewrite at or below {(3 * word_count) // 2} words.\n\n"
                f"Context:\n{chr(10).join(context)}\n\nOutput:"
            )
            rows.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "target_clause": target,
            })
        return rows

    examples = measure_setup("prompt_and_dataset_preparation", prepare_examples)
    if not examples:
        raise ValueError("No valid examples found in the preprocessed dataset")
    dataset = Dataset.from_list(examples)
    print(f"Profiling dataset: {len(dataset)} examples")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The policy model is configured for 4-bit CUDA loading, but CUDA is unavailable. "
            "Run this profiler on the same CUDA environment as GRPO training."
        )
    torch.cuda.reset_peak_memory_stats()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = measure_setup(
        "policy_model_load",
        lambda: AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    reward_device = args.device
    if reward_device == "auto":
        reward_device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = measure_setup(
        "reward_encoder_and_ranker_load",
        lambda: ProfileRewardScorer(checkpoint_path=ranker_path, device=reward_device),
    )

    # Inclusive wall-clock timing for reward phases; the Chrome trace supplies
    # CPU/CUDA operator-level timing for the profiled training steps.
    scorer.cache_originals = timed_call(
        reward_times, "original_cache_lookup_and_cold_encoding", scorer.cache_originals
    )
    scorer.encoder.encode = timed_call(
        reward_times, "sentence_encoder_encode", scorer.encoder.encode
    )
    scorer.ranker.forward = timed_call(
        reward_times, "memory_ranker_forward", scorer.ranker.forward
    )
    reward_function = timed_call(
        reward_times,
        "complete_reward_callback",
        make_reward_function(scorer),
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training_args = GRPOConfig(
        output_dir=str(args.output_dir / "trainer-output"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=2,
        learning_rate=5e-6,
        num_train_epochs=1,
        max_steps=args.steps + 1,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        max_completion_length=48,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        report_to="none",
        remove_unused_columns=False,
    )

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.steps, repeat=1),
        on_trace_ready=lambda active_profiler: active_profiler.export_chrome_trace(str(trace_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            callbacks=[StepTimingCallback(profiler, step_times)],
        )
        print(f"Running 1 warm-up + {args.steps} profiled optimizer steps...")
        trainer.train()
        sync_cuda()

    tables = []
    for sort_key in ("self_cuda_time_total", "self_cpu_time_total"):
        try:
            tables.append(
                f"Top operations by {sort_key}:\n"
                + profiler.key_averages().table(sort_by=sort_key, row_limit=30)
            )
        except (RuntimeError, AssertionError):
            pass
    text_path.write_text("\n\n".join(tables), encoding="utf-8")

    measured_step_times = step_times[1:]
    reward_summary = {
        name: {
            "calls": len(durations),
            "total_seconds": sum(durations),
            "mean_seconds": sum(durations) / len(durations) if durations else 0.0,
        }
        for name, durations in reward_times.items()
    }
    report = {
        "device": str(scorer.device),
        "policy_model": str(model_path),
        "reward_encoder": "sentence-transformers/all-mpnet-base-v2",
        "examples": len(dataset),
        "warmup_steps": 1,
        "profiled_steps_requested": args.steps,
        "completed_optimizer_steps": len(step_times),
        "setup_seconds": setup_seconds,
        "warmup_optimizer_step_seconds": step_times[:1],
        "optimizer_step_seconds": measured_step_times,
        "optimizer_step_mean_seconds": (
            sum(measured_step_times) / len(measured_step_times) if measured_step_times else None
        ),
        "optimizer_step_min_seconds": min(measured_step_times) if measured_step_times else None,
        "optimizer_step_max_seconds": max(measured_step_times) if measured_step_times else None,
        "reward_stage_timings": reward_summary,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trace_file": str(trace_path),
        "operator_summary_file": str(text_path),
        "notes": [
            "Profiling adds overhead; use this run to compare stage cost, not as final throughput.",
            "Reward timing buckets are inclusive and may overlap; do not sum them as exclusive costs.",
            "Optimizer-step timing includes generation, reward evaluation, and training work.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nProfiling complete.")
    print(f"Summary: {report_path}")
    print(f"Chrome trace: {trace_path}")
    print(f"Operator summary: {text_path}")
    if measured_step_times:
        print(f"Mean optimizer step: {report['optimizer_step_mean_seconds']:.2f}s")
    for name, values in reward_summary.items():
        print(f"Reward {name}: {values['total_seconds']:.2f}s total ({values['calls']} calls)")


if __name__ == "__main__":
    main()
