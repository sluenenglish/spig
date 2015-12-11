from unittest import TestCase
import spig
from pandas.util.testing import assert_frame_equal

import pandas as pd


class TestTwoLevelModel(TestCase):
    """Test case for a TwoLevelModel class"""


    def setUp(self):

        self.data = pd.DataFrame([{'grouping' : 'A', 'measure' : 2},
                                  {'grouping' : 'B', 'measure' : 3},
                                  {'grouping' : 'A', 'measure' : 5},
                                  {'grouping' : 'B', 'measure' : 5},
                                  {'grouping' : 'C', 'measure' : 2}])
        self.model = spig.TwoLevelModel(self.data, 'measure', 'grouping')


    def test_init(self):

        assert_frame_equal(self.model.data, self.data)
        self.assertEqual(self.model.measure, 'measure')
        self.assertEqual(self.model.grouping, 'grouping')


    def test_estimate_parameters(self):

        self.model.estimate_parameters()

        self.assertEqual(self.model.Y_stats.shape[0] + self.model.omitted_groups.shape[0],
                         self.data.groupby('grouping').ngroups)


    def test_plot(self):

        self.model.estimate_parameters()
        self.model.random_effects_comparison()
        self.model.plot()
