from collections import defaultdict
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

##################################################################################################################################
# Metrics and confusion matrices are calulated for each individual fold, overall metrics can be calculated by running Metrics.py #
##################################################################################################################################

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/Audio-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"
output_file_path = "C:/Users/44746/Desktop/Project/u_rt_predictions.txt"

try:
     # Loading the fold information from the CSV file
    fold_info = pd.read_csv(fold_csv_path)
    # Creating a list of files for each fold
    fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]
except pd.errors.EmptyDataError:
    print("Error: CSV file is empty")
except pd.errors.ParserError:
    print("Error: CSV file is corrupt or unreadable")
except Exception as e:
    print(f"Unexpected error loading CSV file: {e}")

def load_features_for_fold(directory, filenames):
    segments = []
    labels = []
    identifiers = []
    epsilon = 1e-8  # Small constant to prevent division by zero

    for identifier in filenames:
        filename = f"{identifier}_rt_audio_features.csv"
        path = os.path.join(directory, filename)
        #print(f"Checking path: {path}")  # Debug: print path check
        if not os.path.exists(path):
            print(f"Warning: File does not exist {path}")
            continue

        try:
            features = pd.read_csv(path).values
            #print(f"Features loaded, shape: {features.shape}")  # Debug: print feature shape
            if features.size == 0:
                print("Warning: CSV file is empty")
                continue

            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64
            for start in range(0, len(features) - L + 1, step):
                if len(features) - start >= L:  # Ensure there's enough data for a full segment
                    segment = features[start:start+L]
                    std_dev = np.std(segment, axis=0)
                    zero_variance = std_dev == 0
                    std_dev[zero_variance] = 1
                    normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)
                    segments.append(normalized_segment)
                    labels.append(label)
                    identifiers.append(identifier)
                    #print(f"Segment {len(segments)}: Start at {start}")  # Debug output
        except pd.errors.ParserError:
            print(f"Error: CSV file is corrupt or unreadable at {path}")
            continue

    #print(f"Total segments processed: {len(segments)}")  # Debug: total segments
    return np.array(segments), np.array(labels), identifiers


class CNNRNNModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_layers, bidirectional):
        super(CNNRNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional 
        cnn_output_size = 64 * 64
        self.lstm = nn.LSTM(cnn_output_size, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.attention_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 1, -1)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        attention_weights = F.softmax(self.attention_layer(lstm_out).squeeze(2), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        out = self.fc(context_vector)
        out = self.sigmoid(out)
        return out

def majority_voting(predictions, identifiers):
    if not isinstance(predictions, list) or not isinstance(identifiers, list):
        raise TypeError("Predictions and identifiers must both be lists.")

    if len(predictions) != len(identifiers):
        raise ValueError("Predictions and identifiers must have the same length.")

    vote_counts = defaultdict(list)
    for identifier, prediction in zip(identifiers, predictions):
        if not isinstance(identifier, str):
            raise TypeError("Identifiers must be strings.")
        if not isinstance(prediction, (int, float)):
            raise TypeError("Predictions must be integers or floats.")

        # Here we round floats to the nearest integer.
        prediction = round(prediction) if isinstance(prediction, float) else prediction
        vote_counts[identifier].append(prediction)

    results = {}
    for identifier, votes in vote_counts.items():
        if np.bincount(votes).size > 1:
            results[identifier] = np.argmax(np.bincount(votes))
        else:
            results[identifier] = votes[0] if votes else 0  # Handle empty votes case

    return results

def extract_true_labels(identifiers):
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in identifiers}

all_y_trues, all_y_preds = [], []
temp, temp1 = [], []

for fold_idx in range(len(fold_files)):
    fold_accuracy = 0
    fold_f1 = 0
    fold_precision = 0
    fold_recall = 0
    conf_matrix = confusion_matrix(temp, temp1)

    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)

    # Add random noise
    noise_factor = 0.005
    X_train += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    model = CNNRNNModel(num_channels=32, hidden_dim=128, num_layers=2, bidirectional=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model.train()
    for epoch in range(10):  # 10 for RT, 15 for IT
        for inputs, labels in train_loader:
            try:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error during training: {e}")
        scheduler.step()

    model.eval()
    try:
        with torch.no_grad():
            outputs = model(X_test_tensor).squeeze().numpy()
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        outputs = np.array([])

    predictions = outputs

    file_predictions = {}
    for identifier, prediction in zip(test_identifiers, predictions):
        if identifier not in file_predictions:
            file_predictions[identifier] = []
        file_predictions[identifier].append(prediction)

    file_avg_probability = {filename: np.mean(probs) for filename, probs in file_predictions.items()}

    try:
        with open(output_file_path, 'a') as file:
            for filename, avg_prob in file_avg_probability.items():
                predicted_label = '1' if avg_prob > 0.5 else '0'
                print(f"Filename: {filename}, Average Probability: {avg_prob:.4f}, Predicted Label: {predicted_label}")
                file.write(f"{filename}, {predicted_label}, {avg_prob}\n")
    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}")

    true_labels_dict = extract_true_labels(test_identifiers)
    all_y_trues = [true_labels_dict[identifier] for identifier in set(test_identifiers)]
    all_y_preds = [round(file_avg_probability[identifier]) for identifier in set(test_identifiers)]

    fold_accuracy = accuracy_score(all_y_trues, all_y_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_y_trues, all_y_preds, average='binary')
    precision *= 100
    recall *= 100
    f1 *= 100

    print(f"Fold Metrics: Accuracy: {fold_accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")


    conf_matrix = confusion_matrix(all_y_trues, all_y_preds)
    print("Confusion Matrix:")
    print(conf_matrix)
