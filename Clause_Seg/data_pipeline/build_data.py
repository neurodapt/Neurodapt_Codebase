import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

PROJECT_DIR = Path(__file__).resolve().parent
file_path = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_train.conllu"
print(f"Project Directory: {PROJECT_DIR}")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def generate_sentence_and_boundaries(i,dataset):
    sentence = dataset[i]["sentence"]
    clauses = dataset[i]["clauses"]
    string_sentence = " ".join(sentence)
    clause_ends = []
    current_text = ""
    for clause in clauses:
        clause_string = " ".join(clause)
        if current_text:
            current_text += " "
        current_text += clause_string
        clause_ends.append(len(current_text))
    assert current_text == string_sentence
    encoding = tokenizer(
        string_sentence,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    boundaries = [0] * len(tokens)
    for clause_end in clause_ends[:-1]:
        found = False
        for j in range(len(offsets) - 1):
            current_end = offsets[j][1]
            next_start = offsets[j + 1][0]
            if current_end <= clause_end <= next_start:
                boundaries[j] = 1
                found = True
                break

        if not found:
            print("Could not align boundary:",clause_end)
    return string_sentence, boundaries

def build_dataset():
    #load training_tokens.json
    with open(PROJECT_DIR / "eng.hca.amr" / "training_tokens.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    #build dataset with sentences and boundaries
    dataset = []
    for i in range(len(train_data)):
        sentence, boundaries = generate_sentence_and_boundaries(i, train_data)
        dataset.append({"sentence": sentence,"boundaries": boundaries})

    #build the file train_dataset.json
    with open(PROJECT_DIR / "Dataset" / "train_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


#===================================================================================================
        
    #load validation_tokens.json
    with open(PROJECT_DIR / "eng.hca.amr" / "validation_tokens.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    #build dataset with sentences and boundaries
    dataset = []
    for i in range(len(val_data)):
        sentence, boundaries = generate_sentence_and_boundaries(i, val_data)
        dataset.append({
            "sentence": sentence,
            "boundaries": boundaries
        })

    #build the file validation_dataset.json
    with open(PROJECT_DIR / "Dataset" / "validation_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

#===================================================================================================

    #load test_tokens.json
    with open(PROJECT_DIR / "eng.hca.amr" / "test_tokens.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    #build dataset with sentences and boundaries
    dataset = []
    for i in range(len(test_data)):
        sentence, boundaries = generate_sentence_and_boundaries(i, test_data)
        dataset.append({
            "sentence": sentence,
            "boundaries": boundaries
        })

    #build the file test_dataset.json
    with open(PROJECT_DIR / "Dataset" / "test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    build_dataset()