import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


#Dataset for the precomputed embeddings saved in data_pipeline/optimized_data/*.pt
#each item is: {"text": str, "embeddings": [n, 1536] tensor, "boundaries": list[int]}
class OptimizedClauseBoundaryDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return (
            item["embeddings"],
            torch.tensor(item["boundaries"], dtype=torch.long)
        )


#collate fn for the optimized dataset
#pads both embeddings and boundaries to the max length of the batch
#returns: {"embeddings": [batch, max_tokens, 1536], "boundaries": [batch, max_tokens]}
def optimized_padding(batch):

    embeddings, boundaries = zip(*batch)

    # [batch, max_tokens, 1536]
    embeddings = pad_sequence(embeddings,batch_first=True,padding_value=0.0)

    # [batch, max_tokens]
    boundaries = pad_sequence(boundaries,batch_first=True,padding_value=-100)

    return {"embeddings": embeddings,"boundaries": boundaries} 