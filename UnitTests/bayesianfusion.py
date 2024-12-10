import pytest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
sys.path.append('c:/Users/44746/Desktop/Project') 
from DecisionLevelFusion.Bayesian import decision_level_fusion

@pytest.fixture
def setup_fusion_data():
    """Fixture to provide data for each test."""
    return {
        'empty': ({}, {}, {}, {}, 0.5, {}),
        'audio_only': ({'001': 1}, {}, {'001': 0.7}, {}, 0.5, {}),
        'text_only': ({}, {'001': 0}, {}, {'001': 0.2}, 0.5, {}),
        'matching_predictions': ({'001': 1}, {'001': 1}, {'001': 0.8}, {'001': 0.85}, 0.5, {'001': 1}),
        'non_matching_predictions': ({'001': 0}, {'001': 1}, {'001': 0.4}, {'001': 0.6}, 0.5, {'001': 1}),
        'mismatched_identifiers': ({'001': 1}, {'002': 1}, {'001': 0.8}, {'002': 0.85}, 0.5, {}),
        'invalid_predictions': ({'001': 2}, {'001': -1}, {'001': 0.8}, {'001': 0.85}, 0.5, {}),
        'negative_probabilities': ({'001': 1}, {'001': 1}, {'001': -0.1}, {'001': 0.85}, 0.5, {}),
        'probabilities_over_one': ({'001': 1}, {'001': 1}, {'001': 1.2}, {'001': 0.85}, 0.5, {}),
        'non_numeric_probabilities': ({'001': 1}, {'001': 1}, {'001': 'high'}, {'001': 0.85}, 0.5, {})
    }

def test_empty_inputs(setup_fusion_data):
    data = setup_fusion_data['empty']
    assert decision_level_fusion(*data[:-1]) == data[-1]

def test_audio_present_text_absent(setup_fusion_data):
    data = setup_fusion_data['audio_only']
    assert decision_level_fusion(*data[:-1]) == data[-1]

def test_text_present_audio_absent(setup_fusion_data):
    data = setup_fusion_data['text_only']
    assert decision_level_fusion(*data[:-1]) == data[-1]

def test_matching_predictions(setup_fusion_data):
    data = setup_fusion_data['matching_predictions']
    assert decision_level_fusion(*data[:-1]) == data[-1]

def test_non_matching_predictions_fusion(setup_fusion_data):
    data = setup_fusion_data['non_matching_predictions']
    assert decision_level_fusion(*data[:-1]) == data[-1]

def test_mismatched_identifiers(setup_fusion_data):
    data = setup_fusion_data['mismatched_identifiers']
    result = decision_level_fusion(*data[:-1])
    expected = data[-1]  # Assuming this contains the correctly processed entries without errors.
    assert result == expected, "Mismatched identifiers should be handled or skipped gracefully."

def test_invalid_prediction_values(setup_fusion_data):
    data = setup_fusion_data['invalid_predictions']
    with pytest.raises(ValueError, match="Predictions must be binary \(0 or 1\)"):
        decision_level_fusion(*data[:-1])

def test_negative_probabilities(setup_fusion_data):
    data = setup_fusion_data['negative_probabilities']
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        decision_level_fusion(*data[:-1])

def test_probabilities_over_one(setup_fusion_data):
    data = setup_fusion_data['probabilities_over_one']
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        decision_level_fusion(*data[:-1])

def test_non_numeric_probabilities(setup_fusion_data):
    data = setup_fusion_data['non_numeric_probabilities']
    with pytest.raises(TypeError, match="Probabilities must be numeric"):
        decision_level_fusion(*data[:-1])
