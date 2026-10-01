import torch


def update_counts(logits, targets, valid_mask, threshold=0.5):

    predictions = (torch.sigmoid(logits) > threshold).long()

    targets = targets.long()

    #True Positives
    tp = ((predictions == 1)& (targets == 1)& valid_mask).sum().item()

    #False Positives
    fp = ((predictions == 1)& (targets == 0)& valid_mask).sum().item()

    #False Negatives
    fn = ((predictions == 0)& (targets == 1)& valid_mask).sum().item()

    #Correct Predictions
    correct = (predictions[valid_mask]== targets[valid_mask]).sum().item()

    total = valid_mask.sum().item()

    return tp, fp, fn, correct, total


def calculate_metrics(tp, fp, fn, correct, total):

    #Precision = True Positives / (True Positives + False Positives)
    precision = tp / (tp + fp + 1e-8)

    #recall = True Positives / (True Positives + False Negatives)
    recall = tp / (tp + fn + 1e-8)

    #F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = (2 * precision * recall/ (precision + recall + 1e-8))

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }