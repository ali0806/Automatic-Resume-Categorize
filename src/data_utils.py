import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def load_data(train_path, val_path, test_path):
    """
    Load and preprocess datasets from CSV files.

    Args:
        train_path (str): Path to the training dataset CSV file.
        val_path (str): Path to the validation dataset CSV file.
        test_path (str): Path to the test dataset CSV file.

    Returns:
        tuple: A tuple containing three DataFrames (train_data, val_data, test_data).
    """
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    train_data.dropna(inplace=True)
    val_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    return train_data, val_data, test_data


def generate_dataloader(df, tokenizer, max_length, batch_size):
    """
    Generate a DataLoader for training, validation, or testing.

    Args:
        df (pd.DataFrame): DataFrame containing the text data and labels.
        tokenizer (BertTokenizer): BERT tokenizer for encoding the text.
        max_length (int): Maximum length of the tokenized sequences.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the input dataset.
    """
    ids = np.zeros((len(df), max_length))
    masks = np.zeros((len(df), max_length))
    labels = df['Category'].values

    for i, text in tqdm(enumerate(df['Resume_Text'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask

    X_ids = torch.tensor(ids, dtype=torch.long)
    X_masks = torch.tensor(masks, dtype=torch.long)
    Y_labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(X_ids, X_masks, Y_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
