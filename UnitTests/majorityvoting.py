import unittest
import pytest
import sys
sys.path.append('c:/Users/44746/Desktop/Project')  
from AudioModels.HybridCNNLSTM import majority_voting

def test_empty_input():
    assert majority_voting([], []) == {}

def test_single_element_input():
    assert majority_voting([1], ['A']) == {'A': 1}
    assert majority_voting([0], ['A']) == {'A': 0}

def test_all_same_prediction():
    predictions = [1, 1, 1]
    identifiers = ['A', 'A', 'A']
    assert majority_voting(predictions, identifiers) == {'A': 1}

def test_even_number_of_predictions():
    predictions = [1, 0, 1, 0]
    identifiers = ['A', 'A', 'A', 'A']
    assert majority_voting(predictions, identifiers) == {'A': 0}

def test_multiple_identifiers():
    predictions = [1, 0, 1, 0, 1]
    identifiers = ['A', 'A', 'B', 'B', 'C']
    assert majority_voting(predictions, identifiers) == {'A': 0, 'B': 0, 'C': 1} 

def test_uneven_identifiers():
    predictions = [1, 0, 1]
    identifiers = ['A', 'B', 'C']
    assert majority_voting(predictions, identifiers) == {'A': 1, 'B': 0, 'C': 1}

def test_even_split_handling():
    predictions = [1, 1, 0, 0]
    identifiers = ['A', 'A', 'A', 'A']
    # Depending on expected behavior, define the test's expected result
    assert majority_voting(predictions, identifiers) == {'A': 1}  # or {'A': 0} if you expect rounding down

def test_invalid_inputs():
    with pytest.raises(TypeError):
        majority_voting(None, ['A'])
    with pytest.raises(TypeError):
        majority_voting([1], None)

def test_large_input():
    predictions = [1] * 5000 + [0] * 5000
    identifiers = ['A'] * 5000 + ['B'] * 5000
    assert majority_voting(predictions, identifiers) == {'A': 1, 'B': 0}

def test_even_split_handling():
    predictions = [1, 1, 0, 0]
    identifiers = ['A', 'A', 'A', 'A']
    # Clarify how ties should be resolved, here assuming the last seen value in ties
    expected = {'A': 0}
    assert majority_voting(predictions, identifiers) == expected

def test_single_identifier_different_predictions():
    predictions = [0, 1, 0, 1, 1]
    identifiers = ['A', 'A', 'A', 'A', 'A']
    assert majority_voting(predictions, identifiers) == {'A': 1}

   

if __name__ == '__main__':
    unittest.main()