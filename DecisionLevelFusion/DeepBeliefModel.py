import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from torch.utils.data import TensorDataset, DataLoader

# Load and prepare data function as provided
def load_and_prepare_data(audio_file, text_file):
    def read_file(filename):
        data = {}
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                identifier = parts[0]
                probability = float(parts[2])
                data[identifier] = probability
        return data

    audio_data = read_file(audio_file)
    text_data = read_file(text_file)

    X = []
    y = []
    identifiers = []
    for identifier in audio_data:
        if identifier in text_data:
            audio_prob = audio_data[identifier]
            text_prob = text_data[identifier]
            X.append([audio_prob, text_prob])
            true_label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            y.append(true_label)
            identifiers.append(identifier)
    return np.array(X), np.array(y), identifiers

# Define the DBN like model
class DBNLikeModel(nn.Module):
    def __init__(self):
        super(DBNLikeModel, self).__init__()
        self.fc1 = nn.Linear(2, 256)  # Assuming input features are two (audio_prob, text_prob)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_and_evaluate(X, y, identifiers):
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, identifiers, test_size=0.15, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # DataLoader for batches
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    
    # Model
    model = DBNLikeModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(30):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions.numpy() > 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save predictions with identifiers to a text file
    with open('fused_predictions_dbn_rt.txt', 'w') as file:
        for id, prediction in zip(ids_test, predictions):
            file.write(f"{id},{prediction[0]}\n")

if __name__ == '__main__':
    X, y, identifiers = load_and_prepare_data('audio_rt_predictions.txt', 'text_rt_predictions.txt')
    train_and_evaluate(X, y, identifiers)
