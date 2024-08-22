import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from data_utils import load_data, generate_dataloader
from model import BertResumeClassifier
from train import train_model
from evaluation import evaluate_model
from visualization import plot_accuracy, plot_loss
import json
from pathlib import Path
import argparse
import os

BASE_DIR= Path(__file__).resolve().parent.parent
# Argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate a BERT-based resume classifier.")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation.')
parser.add_argument('--max_length', type=int, default=256, help='Maximum length of input sequences.')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')

args = parser.parse_args()

# Hyperparameters and settings from arguments
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
DROPOUT = 0.3
L2_REG = 1e-5
WEIGHT_DECAY = 1e-6

# Paths to datasets
TRAIN_PATH = f'{BASE_DIR}/data/processed/train_aug.csv'
VAL_PATH = f'{BASE_DIR}/data/processed/val.csv'
TEST_PATH = f'{BASE_DIR}/data/processed/test.csv'


# Load data
train_df, val_df, test_df = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
train_df['Category'] = label_encoder.fit_transform(train_df['Category'])
val_df['Category'] = label_encoder.transform(val_df['Category'])
test_df['Category'] = label_encoder.transform(test_df['Category'])
NUM_CLASSES = len(label_encoder.classes_)

mapping = {category: idx for idx, category in enumerate(label_encoder.classes_)}

# Save mapping to JSON
with open(f'{BASE_DIR}/model/mapping.json', 'w') as f:
    json.dump(mapping, f, indent=4)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Generate dataloaders
train_loader = generate_dataloader(train_df, tokenizer, MAX_LENGTH, BATCH_SIZE)
val_loader = generate_dataloader(val_df, tokenizer, MAX_LENGTH, BATCH_SIZE)
test_loader = generate_dataloader(test_df, tokenizer, MAX_LENGTH, BATCH_SIZE)

# Initialize BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize custom model
resume_classifier = BertResumeClassifier(bert_model, NUM_CLASSES, DROPOUT, L2_REG)

# Check for GPU availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
resume_classifier.to(device)

# Calculate class weights (if needed)
class_counts = train_df['Category'].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)

# Train the model
train_acc, val_acc, train_loss = train_model(
    resume_classifier, 
    train_loader, 
    val_loader, 
    device, 
    EPOCHS, 
    class_weights, 
    LEARNING_RATE, 
    WEIGHT_DECAY, 
    L2_REG
)

# Evaluate the model on the test set
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(resume_classifier, test_loader, device)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Plot training/validation accuracy and loss
plot_accuracy(train_acc, val_acc)
plot_loss(train_loss)
