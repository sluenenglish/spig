"""Test for the generic game class"""
from unittest import TestCase
import spig
from pandas.util.testing import assert_frame_equal

import pandas as pd


class TestTwoLevelModel(TestCase):
    """Test case for a Performance class"""

    def test_init(self):
        data = pd.DataFrame([{'a' : 1, 'b' : 2}, {'a' : 4, 'b' : 2}])
        p = spig.TwoLevelModel(data, 'a', 'b')
        assert_frame_equal(p.data, data)
