#The use of this file is to read files in the Dataset directory and print out different stats regarding them

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
file_path = PROJECT_DIR
print(f"Project Directory: {PROJECT_DIR}")

#1. Length
def print_dataset_lengths():
    print()
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"{file_name} length: {len(data)}")
    print()


#2. Print out the first 3 sentences and their clauses from each dataset
def print_first_three_sentences_and_clauses():
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        print(f"================================  {file_name}  ==============================")
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"First 3 sentences and clauses from {file_name}:")
            for i in range(3):
                print("Sentence:", data[i]["sentence"])
                print("Boundaries:", data[i]["boundaries"])
                print()

#3. boundary distribution for each dataset
def print_boundary_distribution():
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            boundary_counts = {}
            for item in data:
                boundaries = item["boundaries"]
                for b in boundaries:
                    if b not in boundary_counts:
                        boundary_counts[b] = 0
                    boundary_counts[b] += 1
            print(f"Boundary distribution for {file_name}:")
            for b, count in sorted(boundary_counts.items()):
                print(f"Boundary {b}: {count}")
            print(f"Ratio of boundaries to non-boundaries: {boundary_counts.get(1, 0) / boundary_counts.get(0, 1):.4f}%")
            print()

#4. Average sentence length for each dataset (not token wise)
def print_average_sentence_length():
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_length = 0
            for item in data:
                sentence = item["sentence"]
                total_length += len(sentence.split())
            average_length = total_length / len(data)
            print(f"Average sentence length for {file_name}: {average_length:.2f} words")
            print()

#5. Average lenghth of clauses for each dataset (not token wise)
def print_average_clause_length():
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_length = 0
            total_clauses = 0
            for item in data:
                sentence = item["sentence"]
                boundaries = item["boundaries"]
                clause_lengths = []
                current_clause_length = 0
                for i, token in enumerate(sentence.split()):
                    current_clause_length += 1
                    if boundaries[i] == 1:
                        clause_lengths.append(current_clause_length)
                        current_clause_length = 0
                if current_clause_length > 0:
                    clause_lengths.append(current_clause_length)
                total_length += sum(clause_lengths)
                total_clauses += len(clause_lengths)
            average_clause_length = total_length / total_clauses if total_clauses > 0 else 0
            print(f"Average clause length for {file_name}: {average_clause_length:.2f} words")
            print()


#6. no. of clauses per sentence for each dataset
def print_clauses_per_sentence():
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_clauses = 0
            for item in data:
                boundaries = item["boundaries"]
                total_clauses += sum(boundaries)
            average_clauses_per_sentence = total_clauses / len(data) if len(data) > 0 else 0
            print(f"Average number of clauses per sentence for {file_name}: {average_clauses_per_sentence:.2f}")
            print()

#7. print suspecious example (0 boundaries and > 10 boundaries for now) also the number of suspecious examples per dataset at the end
def print_suspicious_examples():
    for file_name in ["train_dataset.json","validation_dataset.json","test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        suspicious_examples = []
        print(f"\n{'=' * 60}")
        print(f"Suspicious examples for {file_name}")
        print(f"{'=' * 60}")
        for i, item in enumerate(data):
            boundaries = item["boundaries"]
            num_boundaries = sum(boundaries)
            # Suspicious condition
            if num_boundaries == 0 or num_boundaries > 10:
                suspicious_examples.append((i, item))
                print(f"\nIndex: {i}")
                print("Sentence:", item["sentence"][:50])
                print("Sentence Length:", len(item["sentence"].split()))
                print("Boundaries:", boundaries[:10])
                print("Number of Boundaries:", num_boundaries)
        print(f"\nTotal suspicious examples in {file_name}: "
              f"{len(suspicious_examples)}")

#8. print clauses smaller than a threshold no. of words with a count of many there are of those in total in train dataset
def print_small_clauses(threshold=2):
    for file_name in ["train_dataset.json", "validation_dataset.json", "test_dataset.json"]:
        with open(PROJECT_DIR / "Dataset" / file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            small_clause_count = 0
            print(f"\n{'=' * 60}")
            print(f"Clauses smaller than {threshold} words in {file_name}")
            print(f"{'=' * 60}")
            for i, item in enumerate(data):
                sentence = item["sentence"]
                boundaries = item["boundaries"]
                current_clause_length = 0
                for j, token in enumerate(sentence.split()):
                    current_clause_length += 1
                    if boundaries[j] == 1:
                        if current_clause_length < threshold:
                            small_clause_count += 1
                            print(f"\nIndex: {i}, Clause Length: {current_clause_length}")
                            print("Clause:", " ".join(sentence.split()[j - current_clause_length + 1:j + 1]))
                        current_clause_length = 0
                if current_clause_length > 0 and current_clause_length < threshold:
                    small_clause_count += 1
                    print(f"\nIndex: {i}, Clause Length: {current_clause_length}")
                    print("Clause:", " ".join(sentence.split()[-current_clause_length:]))
            print(f"\nTotal clauses smaller than {threshold} words in {file_name}: {small_clause_count}")

#9. a graph that shows the distribution of clause lengths in the train dataset, with a bar for each length from 1 to 20, and a bar for all clauses longer than 20
def plot_clause_length_distribution():
    import matplotlib.pyplot as plt
    from collections import defaultdict

    clause_length_distribution = defaultdict(int)

    with open(PROJECT_DIR / "Dataset" / "train_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            sentence = item["sentence"]
            boundaries = item["boundaries"]
            current_clause_length = 0
            for j, token in enumerate(sentence.split()):
                current_clause_length += 1
                if boundaries[j] == 1:
                    clause_length_distribution[current_clause_length] += 1
                    current_clause_length = 0
            if current_clause_length > 0:
                clause_length_distribution[current_clause_length] += 1

    lengths = list(range(1, 21))
    counts = [clause_length_distribution[i] for i in lengths]
    counts.append(sum(count for length, count in clause_length_distribution.items() if length > 20))

    positions = list(range(1, 22))
    labels = [str(i) for i in lengths] + ['>20']
    plt.bar(positions, counts, tick_label=labels)
    plt.xlabel('Clause Length (words)')
    plt.ylabel('Count')
    plt.title('Clause Length Distribution in Train Dataset')
    plt.show()


if __name__ == "__main__":
    print_dataset_lengths()
    #print_first_three_sentences_and_clauses()
    print_boundary_distribution()
    print_average_sentence_length()
    print_average_clause_length()
    #print_clauses_per_sentence()
    #print_suspicious_examples()
    #print_small_clauses()
    plot_clause_length_distribution()