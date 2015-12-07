"""Test for the generic game class"""
from unittest import TestCase
import spig
from pandas.util.testing import assert_frame_equal

import pandas as pd


class TestPerformance(TestCase):
    """Test case for a Performance class"""

    def test_init(self):
        data = pd.DataFrame([{'a' : 1}, {'a' : 4}])
        p = spig.Performance(data)
        assert_frame_equal(p.data, data)
