import argparse
import PyPDF2
import re
from transformers import BertTokenizer , BertModel,BertForSequenceClassification
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import shutil
import pandas as pd
from src.model import BertResumeClassifier
import shutil
import json
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the mapping.json file
with open(f'{BASE_DIR}/model/mapping.json', 'r') as f:
    reverse_mapping = json.load(f)

# Convert the reverse mapping to the desired label_mapping format
label_mapping = {v: k for k, v in reverse_mapping.items()}



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_root = BertModel.from_pretrained('bert-base-cased')


def predict(ids,masks,ckpt):
    model = BertResumeClassifier(model_root,num_classes=24)
    model.to(device)
    model.load_state_dict(torch.load(ckpt,map_location=torch.device('cpu')))
    model.eval()
    # Make predictions
    with torch.no_grad():
        outputs = model(ids, attention_mask=masks)
    prediction = torch.argmax(outputs, dim=1).tolist()
    return prediction
        
    

def text_to_tensor(text):
    # Tokenize the input text
    encoded_input = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Move tensors to the appropriate device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids , attention_mask
        


def preprocessing(resumeText):
    resumeText = resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub('[^a-zA-Z]', ' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main(args):
    filename = []
    category = []
    ROOT = f"{BASE_DIR}/output/categorized_resume"
    model_path = args.ckpt
    file_path = args.file_path
    pdf_files = os.listdir(file_path)
    
    for pdf in tqdm(pdf_files):
        pdf_path = os.path.join(file_path, pdf)
        text = extract_text_from_pdf(pdf_path)
        resume = preprocessing(text)
        ids, masks = text_to_tensor(resume)
        prediction = predict(ids=ids, masks=masks, ckpt=model_path)
        pred_class = label_mapping[prediction[0]]
        
        category_dir = os.path.join(ROOT, pred_class)
        
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        new_file_path = os.path.join(category_dir, pdf)
        
        # Copy the file instead of moving it
        shutil.copy(pdf_path, new_file_path)
        
        filename.append(pdf)
        category.append(pred_class)
    
    categorized_resumes = pd.DataFrame({"filename": filename, "category": category})
    categorized_resumes.to_csv(f"{BASE_DIR}/output/categorized_resumes.csv", index=False)
    print("Prediction completed!")

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument("--ckpt", type=str, default=f"{BASE_DIR}/model/best_model.pt", help="model checkpoint")
    parser.add_argument("--file_path", type=str, help="Resume file path")
    args = parser.parse_args()
    main(args)