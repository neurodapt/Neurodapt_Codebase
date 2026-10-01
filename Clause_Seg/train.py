from tools import *

print("="*20+"Imports"+"="*20)

debug_print("Importing Libraries")

import warnings

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage"
)

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
import tqdm
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import *
from dataset import *
from metrics import *
from losses import *

debug_print("All Import done")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Seed: {SEED}")

#No point running the code without a GPU, change before final publish
try:
    DEVICE = torch.device("cuda")
    debug_print(f"Using GPU: {torch.cuda.get_device_name(DEVICE)}")
except Exception as e:
    debug_print(e)

#Data paths and loaders
PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "train_optimized.pt"
VAL_PATH = PROJECT_DIR / "data_pipeline" / "optimized_data" / "validation_optimized.pt"

train_dataset = OptimizedClauseBoundaryDataset(TRAIN_PATH)
val_dataset = OptimizedClauseBoundaryDataset(VAL_PATH)

MODEL_SAVE_PATH = PROJECT_DIR / "clause_segmentation_model.pt"

# Best model checkpoint path (based on validation F1 score)
BEST_MODEL_PATH = PROJECT_DIR / "clause_segmentation_model_best.pt"
best_f1 = 0.0

#Hyperparameters
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=optimized_padding)

val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,collate_fn=optimized_padding)

debug_print(f"Loaded training samples:{len(train_dataset)} Successfully!")
debug_print(f"Loaded validation samples:{len(val_dataset)} Successfully!")

#Define the model
model = ClauseSegmentationModel().to(DEVICE)

#Loss function is defined in losses.boundary_bce_loss; kept for compatibility notes
optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)

debug_print("Everything defined, now Starting Training!")

print("="*20+"Training"+"="*20)


# Initialize a local best_f1 tracker to avoid UnboundLocalError and reset per weight
best_f1 = 0.0
epochs_without_improvement = 0
for epoch in range(EPOCHS):

    #!!! Training !!!

    # model in train mode
    model.train()
    train_loss = 0.0
    for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} training", leave=False):

        #loading the data
        embeddings = batch["embeddings"].to(DEVICE)
        boundaries = batch["boundaries"].to(DEVICE)

        #padding and masking
        padding_mask = boundaries == -100
        valid_mask = ~padding_mask

        #resetting the gradients
        optimizer.zero_grad()

        #forward pass
        logits = model(embeddings,padding_mask=padding_mask)

        #calculate loss
        loss = weighted_boundary_bce_loss(logits,boundaries,valid_mask, pos_weight=1.0)

        #backward pass and optimization step
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)


    #model in eval mode for proper evaluation (disables dropout, etc.)
    model.eval()

    #Training metrics calculation
    train_tp, train_fp, train_fn, train_correct, train_total = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch in train_loader:
            embeddings = batch["embeddings"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)

            padding_mask = boundaries == -100
            valid_mask = ~padding_mask

            logits = model(embeddings, padding_mask=padding_mask)

            tp, fp, fn, correct, total = update_counts(logits, boundaries, valid_mask)
            train_tp += tp
            train_fp += fp
            train_fn += fn
            train_correct += correct
            train_total += total

    train_metrics = calculate_metrics(train_tp, train_fp, train_fn, train_correct, train_total)
    debug_print(f"[Training] Epoch {epoch+1}/{EPOCHS}  loss={avg_train_loss:.4f}, acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}, Precision={train_metrics['precision']:.4f}, Recall={train_metrics['recall']:.4f}")



    # !!! Validation !!!


    #initialize counts for metrics
    val_loss = 0.0
    val_tp = 0
    val_fp = 0
    val_fn = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:

            #load the data
            embeddings = batch["embeddings"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)

            #padding
            padding_mask = boundaries == -100
            valid_mask = ~padding_mask

            #forward pass
            logits = model(embeddings, padding_mask=padding_mask)

            loss = weighted_boundary_bce_loss(logits, boundaries, valid_mask, pos_weight=1.0)

            #calculate and update counts for metrics
            tp, fp, fn, correct, total = update_counts(logits, boundaries, valid_mask)
            val_tp += tp
            val_fp += fp
            val_fn += fn
            val_correct += correct
            val_total += total
            val_loss += loss.item()

    #calculate metrics and save best model if F1 improves
    val_metrics = calculate_metrics(val_tp, val_fp, val_fn, val_correct, val_total)
    scheduler.step(val_metrics["f1"])

    current_lr = optimizer.param_groups[0]["lr"]
    debug_print(f"[Validation] Epoch {epoch+1}/{EPOCHS} loss={val_loss / len(val_loader):.4f}, acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, lr={current_lr:.2e}")

    #save best model based on validation F1 score
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        debug_print(f"[Validation] New best model saved with F1={best_f1:.4f}")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            debug_print(
                f"[Training] Early stopping after {epoch + 1} epochs; "
                f"validation F1 did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break



print("="*20+"Training Complete"+"="*20)