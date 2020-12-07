import unittest

import torch

from ..utils import TensorList


class TestTensorList(unittest.TestCase):
    """Test cases for TensorList (only the non-trivial methods)."""

    def test_flatten(self):
        t = TensorList(
            [
                torch.Tensor([[1, 2, 3], [3, 2, 1]]),
                torch.Tensor([[4, 5, 6], [6, 5, 4]]),
            ]
        )
        self.assertTrue(
            torch.equal(
                t.flatten(), torch.Tensor([1, 2, 3, 3, 2, 1, 4, 5, 6, 6, 5, 4])
            )
        )
        self.assertTrue(
            torch.equal(
                t.flatten(start_dim=0, end_dim=1),
                torch.Tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6], [6, 5, 4]]),
            )
        )
