from pathlib import Path
from pprint import pprint
import json



PROJECT_DIR = Path(__file__).resolve().parent
file_path = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_train.conllu"
print(f"Project Directory: {PROJECT_DIR}")


#this read and parse the file return a list of tokens representing sentences
#and a list of clauses for each sentence
#each clause is a list of tokens too
def read_conllu(file_path):
    with file_path.open("r", encoding="utf-8") as handle:

        #empty storage for sentences, clauses, and words/tokens
        sentences = []
        sentence = []
        clauses = []
        clause = []

        for line in handle.readlines():#for each line in the file

            #remove whitespace
            line = line.strip()

            if not line:
                if clause:
                    #one clause is done
                    clauses.append(clause)
                if sentence:
                    #one sentence is done
                    sentences.append({
                        "sentence": sentence,
                        "clauses": clauses
                    })
                    sentence = []
                    clauses = []
                    clause = []
                    continue

            #should not be a metadata line
            if line.startswith("#"):
                continue

            #should have all the columns
            columns = line.split("\t")
            if len(columns) != 10:
                continue

            #defining token and boundary information
            token = columns[1]
            misc = columns[9]

            # Add token to sentence
            sentence.append(token)

            # New clause begins
            if "BeginSeg=Yes" in misc:
                # Save previous clause
                if clause:
                    clauses.append(clause)
                # Start new clause
                clause = [token]
            else:
                clause.append(token)

    return sentences


#make 3 .json file from .conllu files in the eng.hca.amr directory
def make_HCA_data():
    train_file = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_train.conllu"
    val_file = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_dev.conllu"
    test_file = PROJECT_DIR / "eng.hca.amr" / "eng.hca.amr_test.conllu"

    train_data = read_conllu(train_file)
    with open(PROJECT_DIR / "eng.hca.amr" / "training_tokens.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    val_data = read_conllu(val_file)
    with open(PROJECT_DIR / "eng.hca.amr" / "validation_tokens.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    test_data = read_conllu(test_file)
    with open(PROJECT_DIR / "eng.hca.amr" / "test_tokens.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)


#Tip : training_tokens.json will not have sytax highlighting since the file is too large for VSCode to handle.
#training_tokens.json size: 15.57 MB
#validation_tokens.json size: 0.72 MB
#test_tokens.json size: 0.76 MB


if __name__ == "__main__":
    make_HCA_data()
