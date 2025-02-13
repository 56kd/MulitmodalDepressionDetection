import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/TF-IDF-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Initialize TF-IDF Vectorizer outside to ensure it is consistent
vectorizer = TfidfVectorizer(max_features=1000)
scaler = StandardScaler()

def prepare_vectorizer(directory, filenames):
    all_texts = []
    for identifier in filenames:
        filename = f"{identifier}_rt_tfidf_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read().replace('\n', ' ')
                all_texts.append(text)
    vectorizer.fit(all_texts)

def load_tfidf_features(directory, filenames):
    all_texts = []
    labels = []
    identifiers = []  # Ensure this list is being filled if needed
    for identifier in filenames:
        filename = f"{identifier}_rt_tfidf_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read().replace('\n', ' ')
                all_texts.append(text)
                label = 1 if 'PM' in identifier or 'PF' in identifier else 0
                labels.append(label)
                identifiers.append(identifier)  # Make sure to append the identifier
        else:
            print(f"File not found: {path}")
    
    features = vectorizer.transform(all_texts).toarray()
    features = scaler.fit_transform(features)
    return np.array(features), np.array(labels), identifiers  # Return identifiers here
# Prepare vectorizer with all text
all_filenames = [file for sublist in fold_files for file in sublist]
prepare_vectorizer(directory, all_filenames)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch, seq_len, hidden_dim]
        # Get the last time step output
        lstm_last_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_last_out)
        x = self.fc(x)
        return x
    
# Load embeddings and labels
all_y_trues, all_y_preds = [], []

for fold_idx in range(len(fold_files)):
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    # Load embeddings and labels
    X_test, y_test, _ = load_tfidf_features(directory, test_filenames)
    X_train, y_train, _ = load_tfidf_features(directory, train_filenames)

    # Convert to PyTorch tensors and reshape for Conv1D: (batch, channels, length)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Verify dimensions
    print("X_train_tensor shape:", X_train_tensor.shape)  # Should be [n_samples, 1, n_features]
    print("X_test_tensor shape:", X_test_tensor.shape)  # Should be [n_samples, 1, n_features]

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    input_dim = X_train_tensor.shape[2]  # Feature dimension of TF-IDF
    hidden_dim = 100  

    model = LSTMModel(input_dim, hidden_dim)

    # Set up weights and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Adjust weights to balance recall between classes
    weights = torch.tensor([2.0], dtype=torch.float32)  # Example adjustment for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    # Training loop
    model.train()
    for epoch in range(100):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().numpy()

    all_y_trues.extend(y_test)
    all_y_preds.extend(predictions)

    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Not Depressed', 'Depressed'], digits=4))

print("Overall Classification Report:")
print(classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed'], digits=4))

overall_accuracy = np.mean(np.array(all_y_trues) == np.array(all_y_preds))
print(f"Overall Accuracy: {overall_accuracy:.4f}")

report = classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed'], digits=4)
print("Overall Classification Report:")
print(report)

precision, recall, f1, _ = precision_recall_fscore_support(all_y_trues, all_y_preds, average='micro')

print("Overall Micro-Average Precision: {:.4f}".format(precision))
print("Overall Micro-Average Recall: {:.4f}".format(recall))
print("Overall Micro-Average F1-Score: {:.4f}".format(f1))
