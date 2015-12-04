"""Test for the generic game class"""
from unittest import TestCase
from spig import Performance

class TestPerformance(TestCase):
    """Test case for a Performance class"""

    def test_init(self):
        data = [1,4,5,6,7]
        p = Performace(data)
        self.assertEqual(p.data, data)
