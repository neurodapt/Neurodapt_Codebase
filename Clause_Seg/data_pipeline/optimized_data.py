#the point of this file is to take the datasets in the dataset folder and generate embeddings for them
#specifically in the concat(token_embeddings + sentence_embeddings) per word for the sentence format, and save them in the optimized_data folder


import torch
import json
import tqdm
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE)
model.eval()

PROJECT_DIR = Path(__file__).resolve().parent

print("Everything defined!")

#tokenize the original text
#embedd the original text
#cat the embeddings of each token with the embeddings of the original text
#return [n,2*embedding_dim] where n is the number of tokens in the original text
def text_to_embeddings(text):
    #tokenize the text
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding.pop("offset_mapping", None) #useless for now(might be useful later)


    encoding = {key: value.to(DEVICE)for key, value in encoding.items()}

    #get the embeddings of the text
    with torch.no_grad():
        outputs = model(**encoding)
    text_embedding = outputs.last_hidden_state.mean(dim=1)  # [1, embedding_dim=768]

    #get the embeddings of each token
    token_embeddings = outputs.last_hidden_state  # [1, n, embedding_dim=768]

    #concatenate the embeddings of each token with the embeddings of the text
    concatenated_embeddings = torch.cat([token_embeddings, text_embedding.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)], dim=2)


    return concatenated_embeddings  # [1, n, 2*embedding_dim=1536]


def build_optimized_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("loading the json file now building...")
    
    optimized_data = []
    #use tqdm to show a progress bar for the data processing
    for item in tqdm.tqdm(data):
        text = item['sentence']
        boundaries = item['boundaries']

        embeddings = text_to_embeddings(text).squeeze(0)  # [n, 1536]
        optimized_data.append({
            'text': text,
            'embeddings': embeddings.cpu(),
            'boundaries': boundaries
        })

    torch.save(optimized_data, output_file)

if __name__ == "__main__":
    print("Starting to optimize the training dataset...")
    input_file = PROJECT_DIR / "Dataset/train_dataset.json"
    output_file = PROJECT_DIR / "optimized_data/train_optimized.pt"
    build_optimized_dataset(input_file, output_file)

    print("Starting to optimize the validation dataset...")
    input_file = PROJECT_DIR / "Dataset/validation_dataset.json"
    output_file = PROJECT_DIR / "optimized_data/validation_optimized.pt"
    build_optimized_dataset(input_file, output_file)

    print("Starting to optimize the test dataset...")
    input_file = PROJECT_DIR / "Dataset/test_dataset.json"
    output_file = PROJECT_DIR / "optimized_data/test_optimized.pt"
    build_optimized_dataset(input_file, output_file)
