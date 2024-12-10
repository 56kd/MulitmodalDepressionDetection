import unittest
import torch
import sys
sys.path.append('c:/Users/44746/Desktop/Project')  
from DecisionLevelFusion.GMU import GatedMultimodalUnit

class TestGatedMultimodalUnit(unittest.TestCase):
    def setUp(self):
        self.model = GatedMultimodalUnit()

    def test_forward_pass_with_valid_input(self):
        input_tensor = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 1))

    def test_forward_pass_with_batch_input(self):
        batch_size = 10
        input_tensor = torch.rand(batch_size, 2, dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_forward_pass_with_zeros(self):
        input_tensor = torch.zeros(1, 2, dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (1, 1))

    def test_forward_pass_with_ones(self):
        input_tensor = torch.ones(1, 2, dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (1, 1))

    def test_forward_pass_with_extreme_values(self):
        input_tensor = torch.tensor([[1000.0, -1000.0]], dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (1, 1))

    def test_forward_pass_with_invalid_shape(self):
        input_tensor = torch.tensor([0.7, 0.3], dtype=torch.float32)  # Missing the extra dimension
        with self.assertRaises(IndexError):
            output = self.model(input_tensor)

    def test_forward_pass_with_large_input(self):
        input_tensor = torch.rand(1000, 2, dtype=torch.float32)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1000, 1))

if __name__ == '__main__':
    unittest.main()