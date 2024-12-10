from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import sys

def load_predictions_and_probabilities(filename):
    predictions = {}
    probabilities = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    identifier, prediction, probability = parts
                    predictions[identifier] = int(float(prediction))
                    probabilities[identifier] = float(probability)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return {}, {}
    except ValueError as e:
        print(f"Error processing file {filename}: {e}")
        return {}, {}
    return predictions, probabilities

def bayesian_fusion(audio_prob, text_prob, prior_depressed):
    epsilon = 1e-10
    try:
        audio_likelihood_ratio = (audio_prob + epsilon) / ((1 - audio_prob) + epsilon)
        text_likelihood_ratio = (text_prob + epsilon) / ((1 - text_prob) + epsilon)
        combined_likelihood_ratio = audio_likelihood_ratio * text_likelihood_ratio
        posterior_prob = (prior_depressed * combined_likelihood_ratio) / (
            (prior_depressed * combined_likelihood_ratio) + (1 - prior_depressed))
    except Exception as e:
        print(f"Numerical stability error in Bayesian fusion: {e}")
        posterior_prob = 0.5
    return posterior_prob

def decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, prior_depressed):
    fused_predictions = {}
    all_identifiers = set(audio_predictions.keys()).union(text_predictions.keys())

    for identifier in all_identifiers:
        audio_pred = audio_predictions.get(identifier)
        text_pred = text_predictions.get(identifier)
        audio_prob = audio_probabilities.get(identifier)
        text_prob = text_probabilities.get(identifier)

        # Skipping processing if one of the predictions is missing
        if audio_pred is None or text_pred is None:
            continue

        # Validate that predictions are integers and probabilities are numeric and within the range [0, 1]
        if not (isinstance(audio_pred, int) and audio_pred in {0, 1}) or not (isinstance(text_pred, int) and text_pred in {0, 1}):
            raise ValueError("Predictions must be binary (0 or 1).")
        if not isinstance(audio_prob, (int, float)) or not isinstance(text_prob, (int, float)):
            raise TypeError("Probabilities must be numeric.")
        if not (0 <= audio_prob <= 1) or not (0 <= text_prob <= 1):
            raise ValueError("Probabilities must be between 0 and 1.")

        # Fuse predictions based on the Bayesian model or simply agree on common predictions
        if audio_pred == text_pred:
            fused_predictions[identifier] = audio_pred
        else:
            fused_prob = bayesian_fusion(audio_prob, text_prob, prior_depressed)
            fused_predictions[identifier] = 1 if fused_prob >= 0.5 else 0

    return fused_predictions

def save_fused_predictions(fused_predictions, filename='fused_predictions_bayes.txt'):
    try:
        with open(filename, 'w') as file:
            for identifier, prediction in fused_predictions.items():
                file.write(f"{identifier},{prediction}\n")
        print("Fused predictions saved to", filename)
    except IOError as e:
        print(f"Failed to save predictions to {filename}: {e}")

def evaluate_predictions(filename='fused_predictions_bayes.txt'):
    y_true = []
    y_pred = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                identifier, prediction = parts
                true_label = 1 if 'PM' in identifier or 'PF' in identifier else 0
                y_true.append(true_label)
                y_pred.append(int(prediction))
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except ValueError as e:
        print(f"Error processing evaluation file {filename}: {e}")

if __name__ == '__main__':
    total_depressed = 64
    total_samples = 116
    prior_depressed = total_depressed / total_samples

    audio_predictions, audio_probabilities = load_predictions_and_probabilities('audio_it_predictions.txt')
    text_predictions, text_probabilities = load_predictions_and_probabilities('text_it_predictions.txt')
    
    try:
        fused_predictions = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, prior_depressed)
        save_fused_predictions(fused_predictions)
        evaluate_predictions()
    except KeyError as e:
        print(f"Key error during fusion: {e}")
    except ValueError as e:
        print(f"Value error during fusion: {e}")
    except TypeError as e:
        print(f"Type error during fusion: {e}")
