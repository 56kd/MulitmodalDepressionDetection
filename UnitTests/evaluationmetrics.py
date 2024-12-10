import os
import unittest
from unittest.mock import patch
import sys
sys.path.append('c:/Users/44746/Desktop/Project')  
from DecisionLevelFusion.WeightedAverage import evaluate_predictions

class TestEvaluatePredictions(unittest.TestCase):
   
    def setUp(self):
        self.test_filename = "test_predictions.txt"
        # Ensure file is clean before each test
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def tearDown(self):
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    @patch('builtins.print')
    def test_evaluate_predictions_happy_case(self, mock_print):
        # Writing data that leads to 100% accuracy
        with open(self.test_filename, "w") as f:
            f.write("PM1,1\n")  # Correct
            f.write("PF2,1\n")  # Correct
            f.write("NM3,0\n")  # Correct
            f.write("NF4,0\n")  # Correct

        evaluate_predictions(self.test_filename)
        
        # Verify that the expected print calls were made
        expected_calls = [
            "Accuracy: 1.0000",
            "Precision: 1.0000",
            "Recall: 1.0000",
            "F1 Score: 1.0000"
        ]
        for call in expected_calls:
            mock_print.assert_any_call(call)

    @patch('builtins.print')
    def test_file_not_found_handling(self, mock_print):
        evaluate_predictions('nonexistent_file.txt')
        mock_print.assert_called_with("Error: File nonexistent_file.txt not found for evaluation.")

    def test_evaluate_predictions_with_incorrect_labels(self):
        # Setup incorrect predictions
        with open(self.test_filename, "w") as f:
            f.write("PM1,0\n")  # Incorrect
            f.write("PF2,0\n")  # Incorrect
            f.write("NM3,1\n")  # Incorrect
            f.write("NF4,1\n")  # Incorrect
        with patch('builtins.print') as mock_print:
            evaluate_predictions(self.test_filename)
            mock_print.assert_any_call("Accuracy: 0.0000")

    def test_evaluate_predictions_mixed_correct_incorrect(self):
        # Half correct, half incorrect setup
        with open(self.test_filename, "w") as f:
            f.write("PM1,1\n")  # Correct
            f.write("PF2,0\n")  # Incorrect
            f.write("NM3,0\n")  # Correct
            f.write("NF4,1\n")  # Incorrect
        with patch('builtins.print') as mock_print:
            evaluate_predictions(self.test_filename)
            mock_print.assert_any_call("Accuracy: 0.5000")

    @patch('builtins.print')
    def test_evaluate_predictions_empty_file(self, mock_print):
        # Ensuring no data is in the file
        evaluate_predictions(self.test_filename)
        mock_print.assert_any_call(f"Error: File {self.test_filename} not found for evaluation.")

    @patch('builtins.print')
    def test_evaluate_predictions_malformed_data(self, mock_print):
        # Create a test file with valid and invalid entries
        test_filename = "test_predictions_malformed.txt"
        with open(test_filename, "w") as f:
            f.write("PM1,1\n")
            f.write("PF2,two\n")  # Malformed, 'two' should be a numeric value
            f.write("NM3,\n")  # Malformed, missing probability value
            f.write("NF4,0.5\n")  # Correct but the probability should be an integer in some contexts

        evaluate_predictions(test_filename)

        # Print all calls to help with debugging
        print(mock_print.call_args_list)

        # Check if any error messages related to malformed data were printed
        found = any("malformed" in str(call).lower() for call in mock_print.call_args_list)
        self.assertTrue(found, "Expected error message for malformed data not found.")

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
