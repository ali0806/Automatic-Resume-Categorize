import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a dataset and compute performance metrics.

    Args:
        model (BertResumeClassifier): The neural network model to be evaluated.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device on which to perform computations (CPU or GPU).

    Returns:
        tuple: Accuracy, precision, recall, and F1 score.
    """
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attn_masks, labels = batch
            input_ids, attn_masks, labels = input_ids.to(device), attn_masks.to(device), labels.to(device)

            outputs = model(input_ids, attn_masks)
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            targets.extend(labels.tolist())

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1
