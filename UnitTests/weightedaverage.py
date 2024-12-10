import sys
import unittest
from unittest.mock import patch

sys.path.append('c:/Users/44746/Desktop/Project')
from DecisionLevelFusion.WeightedAverage import decision_level_fusion

class TestDecisionLevelFusion(unittest.TestCase):

    def test_empty_inputs(self):
        self.assertEqual(decision_level_fusion({}, {}, {}, {}), {})

    def test_unequal_inputs(self):
        audio_predictions = {'id1': 1}
        text_predictions = {}
        audio_probabilities = {'id1': 0.7}
        text_probabilities = {}
        result = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities)
        self.assertEqual(result, {}, "Should handle missing identifiers gracefully without error.")

    def test_single_agreement(self):
        audio_predictions = {'id1': 1}
        text_predictions = {'id1': 1}
        audio_probabilities = {'id1': 0.7}
        text_probabilities = {'id1': 0.8}
        expected = {'id1': 1}
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities), expected)

    def test_single_disagreement_preference_audio(self):
        audio_predictions = {'id1': 1}
        text_predictions = {'id1': 0}
        audio_probabilities = {'id1': 0.8}
        text_probabilities = {'id1': 0.2}
        expected = {'id1': 1}
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=0.6, text_weight=0.4), expected)

    def test_single_disagreement_preference_text(self):
        audio_predictions = {'id1': 0}
        text_predictions = {'id1': 1}
        audio_probabilities = {'id1': 0.3}
        text_probabilities = {'id1': 0.7}
        expected = {'id1': 1}
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=0.4, text_weight=0.6), expected)

    def test_multiple_mixed_predictions(self):
        audio_predictions = {'id1': 1, 'id2': 0}
        text_predictions = {'id1': 1, 'id2': 1}
        audio_probabilities = {'id1': 0.9, 'id2': 0.1}
        text_probabilities = {'id1': 0.85, 'id2': 0.95}
        expected = {'id1': 1, 'id2': 1}
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities), expected)

    def test_weights_impact(self):
        audio_predictions = {'id1': 0, 'id2': 1}
        text_predictions = {'id1': 1, 'id2': 0}
        audio_probabilities = {'id1': 0.1, 'id2': 0.9}
        text_probabilities = {'id1': 0.9, 'id2': 0.1}
        expected_heavy_audio = {'id1': 0, 'id2': 1}
        expected_heavy_text = {'id1': 1, 'id2': 0}
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=0.7, text_weight=0.3), expected_heavy_audio)
        self.assertEqual(decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=0.3, text_weight=0.7), expected_heavy_text)

    def test_invalid_weights(self):
        audio_predictions = {'id1': 1}
        text_predictions = {'id1': 0}
        audio_probabilities = {'id1': 0.8}
        text_probabilities = {'id1': 0.2}
        with self.assertRaises(ValueError):
            decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=-0.1, text_weight=1.1)

    def test_missing_identifier_in_one_input(self):
        audio_predictions = {'id1': 1, 'id2': 1}
        text_predictions = {'id1': 1}
        audio_probabilities = {'id1': 0.9, 'id2': 0.8}
        text_probabilities = {'id1': 0.85}
        expected = {'id1': 1}  # No KeyError, gracefully handle missing identifiers
        result = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities)
        self.assertEqual(result, expected, "Missing identifiers should be handled gracefully.")

if __name__ == '__main__':
    unittest.main()