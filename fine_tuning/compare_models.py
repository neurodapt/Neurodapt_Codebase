from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BASE_MODEL_PATH = HERE / "models" / "Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = HERE / "outputs" / "Qwen2.5-0.5-Instruct-GRPO"
COMPARISON_OUTPUT_DIR = HERE / "outputs" / "Qwen2.5-0.5-Instruct-GRPO" / "comparison"
MEMORY_RANKER_PATH = PROJECT_ROOT / "mem_ranker" / "memory_ranker_model_best.pt"
EMBEDDING_MODEL_PATH = (
    PROJECT_ROOT / "mem_ranker" / "data_pipeline" / "models" / "all_mpnet_base_v2"
)
EMBEDDING_MODEL_ID = os.getenv(
    "MPNET_MODEL_ID", "sentence-transformers/all-mpnet-base-v2"
)
EMBEDDING_DIMENSION = 768

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mem_ranker.model import MemoryRanker

def make_prompt(context: list[str], target: str) -> str:
    target_word_count = len(target.split())
    max_rewrite_words = (3 * target_word_count) // 2
    context_lines = [
        f"[TARGET] {clause}" if clause == target else clause for clause in context
    ]
    return f"""Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below {max_rewrite_words} words (150% of the original's {target_word_count} words).

Context:

{chr(10).join(context_lines)}

Output:"""


PROMPT_CASES = [
    ([
        "The company had struggled for several years.",
        "Its revenue finally began to recover.",
        "The new product became popular with customers.",
        "Sales increased rapidly during the following months.",
    ], "The new product became popular with customers."),
    ([
        "The hikers had been walking since dawn.",
        "The hikers found shelter before the storm arrived.",
        "Rain began to fall across the mountain trail.",
    ], "The hikers found shelter before the storm arrived."),
    ([
        "The museum had been closed for renovations.",
        "The museum opened a new wing last spring.",
        "Visitors can now see the ancient coins on display.",
    ], "The museum opened a new wing last spring."),
    ([
        "Her grandmother wrote to her every winter.",
        "She kept the old letter in a wooden box.",
        "Years later, she found it while cleaning the attic.",
    ], "She kept the old letter in a wooden box."),
    ([
        "The town depended on imported electricity.",
        "The new solar farm began supplying power in June.",
        "Residents noticed fewer outages during the summer.",
    ], "The new solar farm began supplying power in June."),
    ([
        "The nurse reviewed the patients' charts before breakfast.",
        "She changed the afternoon schedule after the consultation.",
        "The ward became quieter later that evening.",
    ], "She changed the afternoon schedule after the consultation."),
    ([
        "Engineers inspected the bridge after the heavy rainfall.",
        "They found a small crack beneath the eastern support.",
        "Traffic was diverted until repairs were completed.",
    ], "They found a small crack beneath the eastern support."),
    ([
        "The teacher introduced the experiment during the morning lesson.",
        "The students recorded the temperature every five minutes.",
        "Their final measurements were close to the predicted values.",
    ], "The students recorded the temperature every five minutes."),
    ([
        "Dark clouds gathered above the valley before noon.",
        "The farmers covered the harvested grain before the rain began.",
        "The fields remained dry through the afternoon.",
    ], "The farmers covered the harvested grain before the rain began."),
    ([
        "The developers reproduced the error on a clean installation.",
        "They fixed the software bug by updating the parser.",
        "The next release passed all of the regression tests.",
    ], "They fixed the software bug by updating the parser."),
    ([
        "The gardener planted herbs beside the kitchen window.",
        "The first shoots appeared after two weeks of sunlight.",
        "She moved the pots indoors when winter arrived.",
    ], "The first shoots appeared after two weeks of sunlight."),
    ([
        "The train was delayed by an obstruction on the track.",
        "Passengers received updated departure times from the conductor.",
        "Most travelers reached the city before midnight.",
    ], "Passengers received updated departure times from the conductor."),
    ([
        "The astronomers prepared the telescope before sunset.",
        "They observed a faint comet near the northern horizon.",
        "The measurements were shared with observatories overseas.",
    ], "They observed a faint comet near the northern horizon."),
    ([
        "The library extended its opening hours during exams.",
        "Students found quiet desks on the upper floor.",
        "The building closed shortly after the final study session.",
    ], "Students found quiet desks on the upper floor."),
    ([
        "The chef tested several recipes for the community dinner.",
        "She selected a lentil soup with roasted vegetables.",
        "Volunteers served the meal in the town hall.",
    ], "She selected a lentil soup with roasted vegetables."),
    ([
        "The river rose quickly after two days of rain.",
        "Emergency workers placed barriers near the low bridge.",
        "Residents moved their cars to higher ground.",
    ], "Emergency workers placed barriers near the low bridge."),
    ([
        "The research team collected soil from three fields.",
        "The samples revealed unusually high nitrogen levels.",
        "Farmers adjusted their fertilizer plans afterward.",
    ], "The samples revealed unusually high nitrogen levels."),
    ([
        "The child practiced the piano every evening.",
        "He performed the difficult piece at the school concert.",
        "His music teacher congratulated him afterward.",
    ], "He performed the difficult piece at the school concert."),
    ([
        "The mechanic examined the car's engine before noon.",
        "She replaced a worn belt near the radiator.",
        "The vehicle passed its inspection that afternoon.",
    ], "She replaced a worn belt near the radiator."),
    ([
        "The village had limited access to clean water.",
        "A new filtration system began operating beside the school.",
        "Children no longer carried water from the distant well.",
    ], "A new filtration system began operating beside the school."),
    ([
        "The journalist interviewed residents throughout the morning.",
        "Her article described the neighborhood's changing character.",
        "The newspaper published the story on its front page.",
    ], "Her article described the neighborhood's changing character."),
    ([
        "The construction crew inspected the foundation carefully.",
        "They discovered moisture beneath the western wall.",
        "The project manager ordered additional drainage work.",
    ], "They discovered moisture beneath the western wall."),
    ([
        "The botanist placed the seedlings under artificial light.",
        "Several plants developed stronger roots within a month.",
        "The results supported the greenhouse experiment.",
    ], "Several plants developed stronger roots within a month."),
    ([
        "The airport announced a delay because of heavy fog.",
        "Passengers waited near the departure gate.",
        "The flight eventually left shortly after sunrise.",
    ], "Passengers waited near the departure gate."),
    ([
        "The coach reviewed the team's defensive formation.",
        "The players practiced quick passes before the match.",
        "Their improved coordination helped them control the game.",
    ], "The players practiced quick passes before the match."),
    ([
        "The doctor examined the patient's injured ankle.",
        "She recommended rest and a light compression bandage.",
        "The swelling decreased over the following week.",
    ], "She recommended rest and a light compression bandage."),
    ([
        "The company moved its records to a secure server.",
        "Employees completed a short security training course.",
        "The information technology team reviewed access logs monthly.",
    ], "Employees completed a short security training course."),
    ([
        "The photographer waited beside the lake at dawn.",
        "She captured the reflection of the mountains in the water.",
        "The image later appeared in a regional magazine.",
    ], "She captured the reflection of the mountains in the water."),
    ([
        "The school replaced its old heating system.",
        "Classrooms remained warmer during the coldest weeks.",
        "The administration reported lower energy use in spring.",
    ], "Classrooms remained warmer during the coldest weeks."),
    ([
        "The archaeologists uncovered a group of clay tablets.",
        "The inscriptions provided evidence of an ancient trade route.",
        "The artifacts were transferred to the regional museum.",
    ], "The inscriptions provided evidence of an ancient trade route."),
    ([
        "The baker prepared the dough before sunrise.",
        "Fresh bread filled the market with a warm aroma.",
        "Customers formed a short line outside the shop.",
    ], "Fresh bread filled the market with a warm aroma."),
    ([
        "The software team planned a maintenance window.",
        "They deployed the database update late on Saturday.",
        "Users received a notice when the service was restored.",
    ], "They deployed the database update late on Saturday."),
    ([
        "The fishermen watched the weather from the harbor.",
        "They postponed the afternoon trip because the waves grew larger.",
        "The boats remained tied to the dock until evening.",
    ], "They postponed the afternoon trip because the waves grew larger."),
    ([
        "The engineer tested the battery under different temperatures.",
        "Its capacity decreased noticeably in the cold chamber.",
        "The team added insulation to the next prototype.",
    ], "Its capacity decreased noticeably in the cold chamber."),
    ([
        "The students compared water samples from two streams.",
        "The downstream sample contained more suspended sediment.",
        "Their report recommended planting vegetation along the banks.",
    ], "The downstream sample contained more suspended sediment."),
    ([
        "The hotel received several reservations for the festival.",
        "Staff prepared additional rooms before the weekend.",
        "Visitors arrived from several nearby cities.",
    ], "Staff prepared additional rooms before the weekend."),
    ([
        "The community garden had little space for new crops.",
        "Volunteers built raised beds along the eastern fence.",
        "The gardeners planted tomatoes and peppers there.",
    ], "Volunteers built raised beds along the eastern fence."),
    ([
        "The teacher noticed that several students were confused.",
        "She explained the equation using a simple diagram.",
        "The class solved the next example without assistance.",
    ], "She explained the equation using a simple diagram."),
    ([
        "The rescue team searched the trail throughout the afternoon.",
        "They located the missing climber near a sheltered ledge.",
        "A helicopter carried him safely to the base camp.",
    ], "They located the missing climber near a sheltered ledge."),
    ([
        "The city planted trees beside the busy highway.",
        "The new vegetation reduced dust near nearby houses.",
        "Residents helped water the young trees during summer.",
    ], "The new vegetation reduced dust near nearby houses."),
    ([
        "The orchestra rehearsed the final movement repeatedly.",
        "The conductor changed the tempo during the quiet passage.",
        "The performance received enthusiastic applause.",
    ], "The conductor changed the tempo during the quiet passage."),
    ([
        "The laboratory freezer stopped working overnight.",
        "Technicians transferred the samples to a backup unit.",
        "The laboratory manager recorded the incident in the log.",
    ], "Technicians transferred the samples to a backup unit."),
    ([
        "The farmer checked the irrigation pipes before planting.",
        "A blocked valve prevented water from reaching one field.",
        "Workers repaired the valve before the next morning.",
    ], "A blocked valve prevented water from reaching one field."),
    ([
        "The museum digitized its collection of historical maps.",
        "Researchers accessed high-resolution images through the website.",
        "The project made fragile documents easier to study.",
    ], "Researchers accessed high-resolution images through the website."),
    ([
        "The cyclist checked the route before leaving the station.",
        "She followed a quiet road through the forest.",
        "The journey ended at a village beside the river.",
    ], "She followed a quiet road through the forest."),
    ([
        "The council discussed several proposals for the empty lot.",
        "Members approved a plan for a public playground.",
        "Construction was scheduled to begin in autumn.",
    ], "Members approved a plan for a public playground."),
    ([
        "The editor reviewed the manuscript over the weekend.",
        "She asked the author to clarify the final paragraph.",
        "The revised version was accepted the following month.",
    ], "She asked the author to clarify the final paragraph."),
    ([
        "The wind damaged several roofs during the night.",
        "Repair crews covered the broken windows with temporary boards.",
        "Insurance inspectors visited the neighborhood the next day.",
    ], "Repair crews covered the broken windows with temporary boards."),
    ([
        "The nutrition team designed a meal plan for athletes.",
        "It included more whole grains and fresh vegetables.",
        "The players followed the plan during preseason training.",
    ], "It included more whole grains and fresh vegetables."),
    ([
        "The theater installed new lights above the stage.",
        "Technicians adjusted the brightness during the dress rehearsal.",
        "The actors could see the set more clearly afterward.",
    ], "Technicians adjusted the brightness during the dress rehearsal."),
    ([
        "The coastal village prepared for the annual festival.",
        "Local artists decorated the main street with colorful banners.",
        "Visitors gathered near the harbor in the evening.",
    ], "Local artists decorated the main street with colorful banners."),
    ([
        "The analyst examined the sales figures from each region.",
        "Online orders accounted for the largest increase.",
        "The company expanded its delivery service afterward.",
    ], "Online orders accounted for the largest increase."),
    ([
        "The volunteers sorted donated clothing by size.",
        "They packed winter coats separately for distribution.",
        "The charity delivered the supplies before the cold weather.",
    ], "They packed winter coats separately for distribution."),
    ([
        "The observatory installed a sensor near the roof.",
        "The instrument measured changes in nighttime air quality.",
        "Scientists compared the readings with data from the valley.",
    ], "The instrument measured changes in nighttime air quality."),
]

DEFAULT_PROMPTS = [make_prompt(context, target) for context, target in PROMPT_CASES]


def load_base_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_context(prompt: str) -> tuple[list[str], int] | None:
    """Read the ordered context clauses and the single [TARGET] clause."""
    lines = prompt.splitlines()
    context_start = next(
        (index for index, line in enumerate(lines) if line.strip().lower() == "context:"),
        None,
    )
    if context_start is None:
        return None

    context_lines = []
    for line in lines[context_start + 1 :]:
        if line.strip().lower() == "output:":
            break
        if line.strip():
            context_lines.append(line.strip())
    else:
        return None

    target_indices = [
        index
        for index, line in enumerate(context_lines)
        if line.startswith("[TARGET]")
    ]
    if len(target_indices) != 1:
        return None

    target_index = target_indices[0]
    context_lines[target_index] = context_lines[target_index][len("[TARGET]") :].strip()
    if not all(context_lines):
        return None
    return context_lines, target_index


class ContextualMemoryRanker:
    """Score candidate rewrites in the surrounding clause sequence."""

    def __init__(self) -> None:
        if not MEMORY_RANKER_PATH.is_file():
            raise FileNotFoundError(f"Memory ranker checkpoint not found: {MEMORY_RANKER_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_source = (
            str(EMBEDDING_MODEL_PATH)
            if EMBEDDING_MODEL_PATH.is_dir()
            else EMBEDDING_MODEL_ID
        )
        if embedding_source == EMBEDDING_MODEL_ID:
            print(
                f"Local MPNet model not found at {EMBEDDING_MODEL_PATH}; "
                f"loading {EMBEDDING_MODEL_ID} from the Hugging Face cache or hub."
            )
        self.embedding_model = SentenceTransformer(
            embedding_source, device=str(self.device)
        )
        self.model = MemoryRanker(input_dim=EMBEDDING_DIMENSION)
        state_dict = torch.load(
            MEMORY_RANKER_PATH, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def score_candidates(
        self, context_clauses: list[str], target_index: int, candidates: list[str]
    ) -> list[float]:
        stories = [
            context_clauses[:target_index] + [candidate] + context_clauses[target_index + 1 :]
            for candidate in candidates
        ]
        flat_clauses = [clause for story in stories for clause in story]
        embeddings = self.embedding_model.encode(
            flat_clauses,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings_tensor = torch.as_tensor(
            embeddings.reshape(len(stories), len(context_clauses), EMBEDDING_DIMENSION),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(embeddings_tensor))[:, target_index]
        return scores.cpu().tolist()


def save_comparison_outputs(
    prompts: list[str],
    base_outputs: list[str],
    tuned_outputs: list[str],
    score_rows: dict[int, tuple[float, float]],
    output_dir: Path,
) -> None:
    """Save per-prompt generations, ranker scores, and diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (prompt, base_output, tuned_output) in enumerate(
        zip(prompts, base_outputs, tuned_outputs), start=1
    ):
        base_score, tuned_score = score_rows.get(index - 1, (None, None))
        rows.append(
            {
                "prompt_index": index,
                "base_output": base_output,
                "tuned_output": tuned_output,
                "base_score": base_score,
                "tuned_score": tuned_score,
                "score_delta": (
                    tuned_score - base_score
                    if base_score is not None and tuned_score is not None
                    else None
                ),
                "base_words": len(base_output.split()),
                "tuned_words": len(tuned_output.split()),
            }
        )

    csv_path = output_dir / "comparison_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scorable_rows = [row for row in rows if row["base_score"] is not None]
    if not scorable_rows:
        print(f"No scorable comparisons; wrote generations to {csv_path}")
        return

    indices = [row["prompt_index"] for row in scorable_rows]
    base_scores = [row["base_score"] for row in scorable_rows]
    tuned_scores = [row["tuned_score"] for row in scorable_rows]
    deltas = [row["score_delta"] for row in scorable_rows]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(indices, base_scores, marker="o", label="Base model")
    axis.plot(indices, tuned_scores, marker="o", label="GRPO adapter")
    axis.set_xlabel("Prompt index")
    axis.set_ylabel("Memory Ranker score")
    axis.set_title("Contextual Memory Ranker Scores")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "memory_ranker_scores.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = ["#2a9d8f" if delta >= 0 else "#e76f51" for delta in deltas]
    axis.bar(indices, deltas, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Prompt index")
    axis.set_ylabel("Adapter score - base score")
    axis.set_title("Change in Contextual Memory Ranker Score")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "memory_ranker_score_delta.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        indices,
        [row["base_words"] for row in scorable_rows],
        marker="o",
        label="Base model words",
    )
    axis.plot(
        indices,
        [row["tuned_words"] for row in scorable_rows],
        marker="o",
        label="GRPO adapter words",
    )
    axis.set_xlabel("Prompt index")
    axis.set_ylabel("Output word count")
    axis.set_title("Base and Adapter Output Lengths")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "output_lengths.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved comparison table and plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the original Qwen model and its trained LoRA adapter."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="User prompt text; repeat this option to compare several prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum generated tokens for each model (default: 24).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARISON_OUTPUT_DIR,
        help="Directory for comparison_results.csv and generated plots.",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not BASE_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {BASE_MODEL_PATH}")
    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER_PATH}; train the model first."
        )

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading the original base model in 4-bit...")
    base_model = load_base_model()
    base_model.eval()
    base_outputs = [
        generate(base_model, tokenizer, prompt, args.max_new_tokens)
        for prompt in tqdm(prompts, desc="Generating base outputs", unit="prompt")
    ]
    release_model(base_model)

    print("Loading the same base model with the trained LoRA adapter...")
    adapted_base = load_base_model()
    tuned_model = PeftModel.from_pretrained(adapted_base, str(ADAPTER_PATH))
    tuned_model.eval()
    tuned_outputs = [
        generate(tuned_model, tokenizer, prompt, args.max_new_tokens)
        for prompt in tqdm(prompts, desc="Generating adapter outputs", unit="prompt")
    ]
    release_model(tuned_model)

    for index, (prompt, base_output, tuned_output) in enumerate(
        zip(prompts, base_outputs, tuned_outputs), start=1
    ):
        print(f"\n{'=' * 80}\nPROMPT {index}\n{'=' * 80}\n{prompt}")
        print(f"\n--- ORIGINAL MODEL ---\n{base_output or '[empty output]'}")
        print(f"\n--- FINE-TUNED MODEL ---\n{tuned_output or '[empty output]'}")

    parsed_contexts = [extract_context(prompt) for prompt in prompts]
    scorable_indices = [
        index
        for index, context in enumerate(parsed_contexts)
        if context is not None and base_outputs[index] and tuned_outputs[index]
    ]
    score_rows: dict[int, tuple[float, float]] = {}
    if scorable_indices:
        print("\nLoading the memory ranker and MPNet encoder...")
        ranker = ContextualMemoryRanker()
        for index in tqdm(scorable_indices, desc="Scoring generated outputs", unit="prompt"):
            context_clauses, target_index = parsed_contexts[index]
            scores = ranker.score_candidates(
                context_clauses,
                target_index,
                [base_outputs[index], tuned_outputs[index]],
            )
            winner = "ORIGINAL MODEL" if scores[0] >= scores[1] else "FINE-TUNED MODEL"
            score_rows[index] = (scores[0], scores[1])
            print(f"\n--- CONTEXTUAL MEMORY RANKING (PROMPT {index + 1}) ---")
            print(f"Original model score:   {scores[0]:.4f}")
            print(f"Fine-tuned model score: {scores[1]:.4f}")
            print(f"Higher-ranked rewrite:  {winner}")

    unscorable_indices = [
        index
        for index, context in enumerate(parsed_contexts)
        if context is None and base_outputs[index] and tuned_outputs[index]
    ]
    for index in unscorable_indices:
        print(
            f"\nSkipping memory ranking for prompt {index + 1}: add a Context: section "
            "with one [TARGET] line and an Output: delimiter."
        )

    save_comparison_outputs(
        prompts,
        base_outputs,
        tuned_outputs,
        score_rows,
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
