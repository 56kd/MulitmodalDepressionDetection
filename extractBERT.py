import os
import subprocess
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel

# Define paths
output_folder = "C:/Users/44746/Desktop/Interplay/BERT-Interview"
data_dir = "C:/Users/44746/Desktop/Project/Interview"

# BERT Model and Tokenizer Initialization
model_name = "neuraly/bert-base-italian-cased-sentiment" # Using an Italian model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)


def extract_text_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs['input_ids'])
    # Using the pooled output for representing the entire sequence
    return outputs.pooler_output.numpy()

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    text_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    for text_file in text_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()
            output_csv_file = os.path.join(output_folder, f"{os.path.basename(text_file).replace('.txt', '_bert_features.csv')}")
            bert_features = extract_text_features(text, tokenizer, model)
            # Save BERT features to CSV
            pd.DataFrame(bert_features).to_csv(output_csv_file, index=False)
        except FileNotFoundError:
            print(f"Text file not found for {text_file}. Skipping.")

if __name__ == "__main__":
    main()

