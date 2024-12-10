import os
import unittest
from unittest.mock import call, patch, mock_open
import sys
sys.path.append('c:/Users/44746/Desktop/Project')  
from DecisionLevelFusion.WeightedAverage import save_fused_predictions  

class TestSaveFusedPredictions(unittest.TestCase):

    def test_save_empty_predictions(self):
        filename = "empty_predictions_test.txt"
        save_fused_predictions({}, filename)
        self.assertTrue(os.path.exists(filename))
        with open(filename, 'r') as file:
            self.assertEqual(file.read(), "")

    def test_save_null_input(self):
        with self.assertRaises(TypeError):
            save_fused_predictions(None)

    @patch("builtins.open", new_callable=mock_open)
    def test_save_happy_case(self, mock_file):
        predictions = {"id1": 1, "id2": 0}
        save_fused_predictions(predictions)
        mock_file().write.assert_any_call("id1,1\n")
        mock_file().write.assert_any_call("id2,0\n")


    @patch("builtins.open", new_callable=mock_open)
    def test_save_with_custom_filename(self, mock_file):
        filename = "custom_filename_test.txt"
        predictions = {"id3": 1}
        save_fused_predictions(predictions, filename)
        mock_file.assert_called_with(filename, 'w')
        mock_file().write.assert_called_once_with('id3,1\n')

    @patch("builtins.open", mock_open())
    def test_mock_save_check_call(self):
        predictions = {"id4": 0}
        filename = "mock_check_call.txt"
        save_fused_predictions(predictions, filename)
        open.assert_called_with(filename, 'w')

    @patch("builtins.open", new_callable=mock_open)
    def test_save_malformed_predictions(self, mock_file):
        # Simulate malformed data: incorrect data types or structures
        malformed_predictions = {"id1": "one", "id2": None}  # assuming predictions should be integers
        with self.assertRaises(ValueError):
            save_fused_predictions(malformed_predictions)
        # Check no write operations occurred due to the malformed data
        mock_file().write.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    def test_save_complex_predictions(self, mock_file):
        predictions = {"id1": 1, "id2": 0, "id3": 1, "id4": 0, "id5": 1}
        save_fused_predictions(predictions)
        calls = [
            call("id1,1\n"),
            call("id2,0\n"),
            call("id3,1\n"),
            call("id4,0\n"),
            call("id5,1\n")
        ]
        mock_file().write.assert_has_calls(calls, any_order=True)

if __name__ == "__main__":
    unittest.main()