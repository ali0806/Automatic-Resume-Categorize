import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def train_model(model, train_loader, val_loader, device, num_epochs, class_weights, learning_rate, weight_decay, l2_reg):
    """
    Train the BERT-based resume classifier model.

    Args:
        model (BertResumeClassifier): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device on which to perform computations (CPU or GPU).
        num_epochs (int): Number of training epochs.
        class_weights (torch.Tensor): Weights for each class used in the loss function.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer (L2 regularization).
        l2_reg (float): L2 regularization factor for the model parameters.

    Returns:
        tuple: Training accuracy, validation accuracy, and training loss over epochs.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)
    
    best_val_accuracy = 0.0
    early_stopping_counter = 0
    train_acc, valid_acc, train_loss = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, predictions, targets = 0.0, [], []
        print(f"---------Epoch: {epoch}----------")
        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids, attn_masks, labels = batch
            input_ids, attn_masks, labels = input_ids.to(device), attn_masks.to(device), labels.to(device)

            outputs = model(input_ids, attn_masks)
            loss = criterion(outputs, labels)
            
            # Apply L2 regularization
            l2_loss = torch.tensor(0.).to(device)
            for param in model.parameters():
                if param.dim() > 1:
                    l2_loss += torch.norm(param, p=2)
            loss += l2_reg * l2_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            targets.extend(labels.tolist())

        avg_loss = total_loss / len(train_loader)
        train_acc.append(accuracy_score(targets, predictions))

        # Validation
        model.eval()
        val_predictions, val_targets = [], []

        with torch.no_grad():
            for val_batch in tqdm(val_loader):
                val_input_ids, val_attn_masks, val_labels = val_batch
                val_input_ids, val_attn_masks, val_labels = val_input_ids.to(device), val_attn_masks.to(device), val_labels.to(device)

                val_outputs = model(val_input_ids, val_attn_masks)
                val_predictions.extend(torch.argmax(val_outputs, dim=1).tolist())
                val_targets.extend(val_labels.tolist())

        val_acc = accuracy_score(val_targets, val_predictions)
        valid_acc.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg. Loss: {avg_loss:.4f} - Train Accuracy: {train_acc[-1]:.4f} - Val Accuracy: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"{BASE_DIR}/model/best_model.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 8:
                print("Early stopping triggered.")
                break

    return train_acc, valid_acc, train_loss
