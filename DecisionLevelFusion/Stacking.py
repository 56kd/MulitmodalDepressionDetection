import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions_and_probabilities(filename):
    """Load predictions and probabilities from a file."""
    predictions = {}
    probabilities = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    identifier, prediction, probability = parts
                    prob = float(probability)
                    if not (0 <= prob <= 1):
                        logging.error(f"Probability out of bounds: {prob} for identifier {identifier}")
                        continue
                    predictions[identifier] = int(float(prediction))
                    probabilities[identifier] = prob
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        raise
    return predictions, probabilities

def train_meta_learner(audio_probabilities, text_probabilities, y_true):
    """
    Train a logistic regression model as the meta-learner.
    
    :param audio_probabilities: Probabilities from the audio model.
    :param text_probabilities: Probabilities from the text model.
    :param y_true: Actual labels.
    :return: Trained logistic regression model.
    """
    # Check if all labels are either 0 or 1
    if any(label not in [0, 1] for label in y_true):
        raise ValueError("Labels must be binary, either 0 or 1.")

    # Validate probabilities are within the [0, 1] range
    all_probabilities = list(audio_probabilities.values()) + list(text_probabilities.values())
    if any(p < 0 or p > 1 for p in all_probabilities):
        raise ValueError("Probabilities must be between 0 and 1.")

    # Prepare the training data
    X = np.vstack((list(audio_probabilities.values()), list(text_probabilities.values()))).T
    y = np.array(y_true)
    
    # Initialize and train the logistic regression model
    meta_learner = LogisticRegression()
    meta_learner.fit(X, y)
    
    return meta_learner

def decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, meta_learner, threshold=0.5):
    fused_predictions = {}
    for identifier in audio_predictions:
        if identifier in text_predictions and identifier in audio_probabilities and identifier in text_probabilities:
            audio_prob = audio_probabilities[identifier]
            text_prob = text_probabilities[identifier]
            fused_prob = meta_learner.predict_proba([[audio_prob, text_prob]])[0, 1]
            fused_prediction = 1 if fused_prob > threshold else 0
            #print(f"ID: {identifier}, Audio Prob: {audio_prob}, Text Prob: {text_prob}, Fused Prob: {fused_prob}, Prediction: {fused_prediction}")
            fused_predictions[identifier] = fused_prediction
    return fused_predictions


def save_fused_predictions(fused_predictions, filename='fused_predictions_lr_rt.txt'):
    """Save the fused predictions to a file."""
    try:
        with open(filename, 'w') as file:
            for identifier, prediction in fused_predictions.items():
                file.write(f"{identifier},{prediction}\n")
        logging.info(f"Fused predictions saved to {filename}")
    except IOError as e:
        logging.error(f"Failed to save predictions: {e}")

def evaluate_predictions(filename='fused_predictions_lr_rt.txt'):
    """Evaluate the predictions against true labels."""
    y_true, y_pred = [], []
    try:
        with open(filename, 'r') as file:
            for line in file:
                identifier, prediction = line.strip().split(',')
                true_label = 1 if 'PM' in identifier or 'PF' in identifier else 0
                y_true.append(true_label)
                y_pred.append(int(prediction))
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    precision = precision_score(y_true, y_pred, zero_division=1)

    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if __name__ == '__main__':
    try:
        # Load predictions and probabilities
        audio_predictions, audio_probabilities = load_predictions_and_probabilities('audio_rt_predictions.txt')
        text_predictions, text_probabilities = load_predictions_and_probabilities('text_rt_predictions.txt')
        
        # Extract true labels for training the meta-learner
        y_true = [1 if 'PM' in id or 'PF' in id else 0 for id in audio_predictions]  # Simplified extraction
        
        # Train the meta-learner with probabilities and actual labels
        meta_learner = train_meta_learner(audio_probabilities, text_probabilities, y_true)
        
        # Perform decision-level fusion using the trained meta-learner
        fused_predictions = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, meta_learner)
        
        # Save and evaluate the fused predictions
        save_fused_predictions(fused_predictions)
        evaluate_predictions()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
