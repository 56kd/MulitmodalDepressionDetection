import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from unittest.mock import MagicMock
import sys
sys.path.append('c:/Users/44746/Desktop/Project') 
from DecisionLevelFusion.Stacking import train_meta_learner

# Assuming the rest of the module's code and imports are as provided in the snippet.

@pytest.fixture
def sample_data():
    audio_probs = {'id1': 0.1, 'id2': 0.4, 'id3': 0.7}
    text_probs = {'id1': 0.2, 'id2': 0.5, 'id3': 0.8}
    y_true = [0, 1, 1]
    return audio_probs, text_probs, y_true

def test_train_meta_learner_with_valid_input(sample_data):
    audio_probs, text_probs, y_true = sample_data
    model = train_meta_learner(audio_probs, text_probs, y_true)
    assert isinstance(model, LogisticRegression), "Model is not an instance of LogisticRegression"
    assert model.classes_.size == 2, "Model does not support binary classification"

def test_train_meta_learner_with_empty_input():
    audio_probs, text_probs, y_true = {}, {}, []
    with pytest.raises(ValueError):
        train_meta_learner(audio_probs, text_probs, y_true)

def test_train_meta_learner_with_mismatched_lengths():
    audio_probs = {'id1': 0.1, 'id2': 0.4}
    text_probs = {'id1': 0.2, 'id2': 0.5, 'id3': 0.8}
    y_true = [0, 1]
    with pytest.raises(ValueError):
        train_meta_learner(audio_probs, text_probs, y_true)

def test_train_meta_learner_with_invalid_probabilities(sample_data):
    audio_probs, text_probs, y_true = sample_data
    audio_probs['id1'] = -0.1  # Set an invalid probability
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        train_meta_learner(audio_probs, text_probs, y_true)

def test_train_meta_learner_with_invalid_labels():
    audio_probs = {'id1': 0.1, 'id2': 0.4}
    text_probs = {'id1': 0.2, 'id2': 0.5}
    y_true = [0, 2]  # Non-binary label
    with pytest.raises(ValueError, match="Labels must be binary, either 0 or 1"):
        train_meta_learner(audio_probs, text_probs, y_true)


def test_train_meta_learner_with_single_element(sample_data):
    audio_probs, text_probs, y_true = sample_data
    # Use single element but ensuring two classes for a valid test
    audio_probs = {'id1': audio_probs['id1'], 'id2': audio_probs['id2']}
    text_probs = {'id1': text_probs['id1'], 'id2': text_probs['id2']}
    y_true = [0, 1]  # Two different class labels
    model = train_meta_learner(audio_probs, text_probs, y_true)
    assert isinstance(model, LogisticRegression), "Single element training failed"

def test_train_meta_learner_with_all_same_label():
    audio_probs = {'id1': 0.1, 'id2': 0.4, 'id3': 0.7}
    text_probs = {'id1': 0.2, 'id2': 0.5, 'id3': 0.8}
    y_true = [1, 1, 1]  # All labels are the same
    with pytest.raises(ValueError):
        train_meta_learner(audio_probs, text_probs, y_true)

def test_train_meta_learner_integration_with_mocked_input():
    # Correct the mock setup to ensure dimension consistency
    np.vstack = MagicMock(return_value=np.array([[0.15, 0.25], [0.45, 0.55], [0.75, 0.85]]).T)  # Correct dimensions
    audio_probs = {'id1': 0.1, 'id2': 0.4, 'id3': 0.7}
    text_probs = {'id1': 0.2, 'id2': 0.5, 'id3': 0.8}
    y_true = [0, 1, 1]
    model = train_meta_learner(audio_probs, text_probs, y_true)
    assert np.vstack.call_count == 1, "numpy.vstack was not called exactly once"