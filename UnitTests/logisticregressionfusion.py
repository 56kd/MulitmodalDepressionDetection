import pytest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
sys.path.append('c:/Users/44746/Desktop/Project') 
from DecisionLevelFusion.Stacking import decision_level_fusion


@pytest.fixture
def setup_decision_level_fusion():
    meta_learner = MagicMock(spec=LogisticRegression)
    # Adjusting the mock to simulate decision boundaries accurately
    def mock_predict_proba(X):
        # Custom logic to return a deterministic output based on input values
        results = []
        for x in X:
            if x[0] > x[1]:
                results.append([0.49, 0.51])  # Class 1 is slightly more probable
            else:
                results.append([0.51, 0.49])  # Class 0 is slightly more probable
        return np.array(results)
    
    meta_learner.predict_proba = MagicMock(side_effect=mock_predict_proba)
    
    return {
        'meta_learner': meta_learner,
        'audio_predictions': {'id1': 1, 'id2': 0},
        'text_predictions': {'id1': 1, 'id2': 0},
        'audio_probabilities': {'id1': 0.8, 'id2': 0.2},
        'text_probabilities': {'id1': 0.7, 'id2': 0.3}
    }

def test_happy_case(setup_decision_level_fusion):
    expected = {'id1': 1, 'id2': 0}
    result = decision_level_fusion(
        setup_decision_level_fusion['audio_predictions'],
        setup_decision_level_fusion['text_predictions'],
        setup_decision_level_fusion['audio_probabilities'],
        setup_decision_level_fusion['text_probabilities'],
        setup_decision_level_fusion['meta_learner']
    )
    assert result == expected, "Happy case failed with unexpected prediction results."

def test_probabilities_boundaries(setup_decision_level_fusion):
    audio_probabilities = {'id1': 0.01, 'id2': 0.99}
    text_probabilities = {'id1': 0.99, 'id2': 0.01}
    expected = {'id1': 0, 'id2': 1}  # Expectations adjusted for the mock's new logic
    result = decision_level_fusion(
        setup_decision_level_fusion['audio_predictions'],
        setup_decision_level_fusion['text_predictions'],
        audio_probabilities,
        text_probabilities,
        setup_decision_level_fusion['meta_learner']
    )
    assert result == expected, "Boundary conditions test failed with unexpected prediction results."
