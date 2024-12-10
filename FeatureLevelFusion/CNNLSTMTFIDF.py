from collections import Counter
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory_audio = "C:/Users/44746/Desktop/Project/Audio-Reading"
directory_text = "C:/Users/44746/Desktop/Project/TF-IDF-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]
output_file_path = "C:/Users/44746/Desktop/Project/concat_rt_predictions_demo.txt"

def apply_pca(features, n_components=100):
    if features.shape[1] < n_components:
        print(f"Reducing PCA components from {n_components} to {features.shape[1]} due to insufficient features")
        n_components = features.shape[1]
    pca = PCA(n_components=n_components)
    transformed_features = pca.fit_transform(features)
    return transformed_features

def load_features_for_fold(directory_audio, directory_text, filenames, n_pca_components=80):
    segments = []
    labels = []
    identifiers = []
    epsilon = 1e-8

    for identifier in filenames:
        audio_filename = f"{identifier}_rt_audio_features.csv"
        text_filename = f"{identifier}_rt_tfidf_features.csv"
        audio_path = os.path.join(directory_audio, audio_filename)
        text_path = os.path.join(directory_text, text_filename)
        
        if os.path.exists(audio_path) and os.path.exists(text_path):
            audio_features = pd.read_csv(audio_path).values
            text_features = pd.read_csv(text_path).values
            if text_features.shape[0] == 1:
                text_features = np.repeat(text_features, audio_features.shape[0], axis=0)

            features = np.concatenate((audio_features, text_features), axis=1)
            features = apply_pca(features, n_pca_components)

            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64

            for start in range(0, len(features) - L + 1, step):
                segment = features[start:start+L]
                if segment.shape[0] == L:
                    std_dev = np.std(segment, axis=0)
                    zero_variance = std_dev == 0
                    std_dev[zero_variance] = 1
                    normalized_segment = (segment - np.mean(segment, axis=0)) / std_dev
                    segments.append(normalized_segment)
                    labels.append(label)
                    identifiers.append(identifier)
                else:
                    print(f"Skipping segment with incorrect shape {segment.shape} at {identifier}")

    try:
        segments = np.array(segments, dtype=np.float32)
    except Exception as e:
        print(f"Error during array conversion: {e}")
        segments = []

    return segments, np.array(labels, dtype=np.int32), identifiers

class CNNRNNModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_layers, bidirectional):
        super(CNNRNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # Example input length is the number of PCA components, adjust accordingly
        input_length = 128  # Assuming 'num_channels' refers to the length after PCA and not 'channels' in the traditional sense
        conv_output_length = (input_length - 5 + 2*2) // 1 + 1
        pooled_output_length = (conv_output_length - 2) // 2 + 1
        cnn_output_size = 64 * pooled_output_length  # This is now the flattened output size after the conv and pool layers

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(cnn_output_size, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * multiplier, 1)
        self.sigmoid = nn.Sigmoid()
        self.attention_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = x.view(x.size(0), 1, -1)  # LSTM expects (batch, seq_len, features)
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

def extract_true_labels(identifiers):
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in identifiers}

all_y_trues, all_y_preds = [], []
temp, temp1 = [], []

for fold_idx in range(len(fold_files)):
    fold_accuracy = 0
    fold_f1 = 0
    fold_precision = 0
    fold_recall = 0

    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    X_test, y_test, test_identifiers = load_features_for_fold(directory_audio, directory_text, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory_audio, directory_text, train_filenames)

    if X_train.ndim == 3:
        X_train = X_train.transpose(0, 2, 1)
    else:
        print("Error: X_train is not a 3D array.")

    X_test = X_test.transpose(0, 2, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    model = CNNRNNModel(num_channels=80, hidden_dim=128, num_layers=2, bidirectional=False)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model.train()
    for epoch in range(5):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor).squeeze().numpy()


    predictions = outputs
    file_predictions = {}
    for identifier, prediction in zip(test_identifiers, predictions):
        if identifier not in file_predictions:
            file_predictions[identifier] = []
        file_predictions[identifier].append(prediction)

    file_avg_probability = {filename: np.mean(probs) for filename, probs in file_predictions.items()}

    with open(output_file_path, 'a') as file:
        for filename, avg_prob in file_avg_probability.items():
            predicted_label = '1' if avg_prob > 0.5 else '0'
            print(f"Filename: {filename}, Predicted Probability: {avg_prob:.4f}, Predicted Label: {predicted_label}")
            file.write(f"{filename}, {predicted_label}, {avg_prob}\n")

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
