import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seeds to ensure reproducibility of results
torch.manual_seed(0)
np.random.seed(0)

# Define paths and load the CSV containing information about the text data folds
directory = "C:/Users/44746/Desktop/Project/Text-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"
try:
    fold_info = pd.read_csv(fold_csv_path)
    fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]
except Exception as e:
    print(f"Failed to load fold data: {e}")

# Load tokenizer and model from Hugging Face's Transformers library
model_name = "neuraly/bert-base-italian-cased-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
except Exception as e:
    print(f"Failed to load model or tokenizer: {e}")

def group_sentences(text):
# Tokenize text into sentences and group them to create longer segments
    try:
        sentences = sent_tokenize(text, language='italian')
        grouped_sentences = [' '.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        return grouped_sentences
    except Exception as e:
        print(f"Failed to group sentences: {e}")
        return []

def load_text_for_fold(directory, filenames):
    # Load and process text data from files, extracting grouped sentences and labels
    texts = []
    labels = []
    identifiers = []
    for identifier in filenames:
        filename = f"{identifier}.txt"
        path = os.path.join(directory, filename)
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                grouped_text = group_sentences(text)
                for group in grouped_text:
                    texts.append(group)
                    label = 1 if 'PM' in identifier.upper() or 'PF' in identifier.upper() else 0
                    labels.append(label)
                    identifiers.append(identifier)
        except Exception as e:
            print(f"Failed to process file {path}: {e}")
    return texts, labels, identifiers

def preprocess_texts_and_create_dataset(texts, labels):
    # Preprocess texts to create a dataset suitable for model training
    try:
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        dataset = Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": torch.tensor(labels)})
        return dataset
    except Exception as e:
        print(f"Failed to preprocess texts and create dataset: {e}")
        return None

def majority_voting(predictions, probabilities, identifiers):
    # Perform majority voting to determine final predictions based on probabilities
    votes = {}
    prob_sum = {}
    for identifier, prediction, probability in zip(identifiers, predictions, probabilities):
        if identifier not in votes:
            votes[identifier] = []
            prob_sum[identifier] = 0.0
        votes[identifier].append(prediction)
        prob_sum[identifier] += probability
    
    final_predictions = {}
    for identifier, votes_list in votes.items():
        avg_prob = prob_sum[identifier] / len(votes_list)
        final_prediction = 1 if avg_prob > 0.5000 else 0
        final_predictions[identifier] = (final_prediction, avg_prob)
    return final_predictions

def save_predictions_to_file(predictions, fold_idx, filename='text_rt_predictions.txt'):
    # Save the predictions to a file
    try:
        with open(filename, 'a') as file:
            for identifier, (prediction, probability) in predictions.items():
                file.write(f"{identifier}, {prediction}, {probability:.17f}\n")
        print(f"Fold {fold_idx} predictions with probabilities appended to {filename}")
    except Exception as e:
        print(f"Failed to save predictions: {e}")

overall_accuracy = []
precisions = []
recalls = []
f1_scores = []

# Process each fold, training the model and evaluating it
for fold_idx in range(len(fold_files)):
    print(f"Processing fold {fold_idx + 1} as test set")
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    train_texts, train_labels, _ = load_text_for_fold(directory, train_filenames)
    test_texts, test_labels, test_identifiers = load_text_for_fold(directory, test_filenames)
    
    train_dataset = preprocess_texts_and_create_dataset(train_texts, train_labels)
    test_dataset = preprocess_texts_and_create_dataset(test_texts, test_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy='epoch',
    )
    
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        trainer.train()
        raw_pred, _, _ = trainer.predict(test_dataset)
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        continue
    
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(torch.tensor(raw_pred)).numpy()[:, 1]
    predictions = np.argmax(raw_pred, axis=1)
    
    voted_predictions = majority_voting(predictions, probabilities, test_identifiers)
    save_predictions_to_file(voted_predictions, fold_idx + 1)
    
    true_labels = {identifier: label for identifier, label in zip(test_identifiers, test_labels)}
    true_labels_list = [true_labels[identifier] for identifier in test_identifiers]
    predictions_list = [voted_predictions[identifier][0] for identifier in test_identifiers]
    
    fold_accuracy = np.mean([true_labels[identifier] == voted_predictions[identifier][0] for identifier in test_identifiers])
    overall_accuracy.append(fold_accuracy)
    
    precision = precision_score(true_labels_list, predictions_list)
    recall = recall_score(true_labels_list, predictions_list)
    f1 = f1_score(true_labels_list, predictions_list)
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

mean_accuracy = np.mean(overall_accuracy)
std_accuracy = np.std(overall_accuracy)
print(f"Overall Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
print(f"Overall Precision: {precisions:.4f}")
print(f"Overall Recall: {recalls:.4f}")
print(f"Overall F1 Score: {f1_scores:.4f}")
