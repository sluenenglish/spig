from unittest import TestCase
import spig
from pandas.util.testing import assert_frame_equal
from hypothesis import given, example, assume
from hypothesis.strategies import text, floats, fixed_dictionaries, composite, lists, sampled_from

import pandas as pd

@composite
def generate_data_and_keys(draw):
    measure = draw(text(min_size=1))
    grouping = draw(text(min_size=1))
    #assume(measure != grouping)
    groups = draw(lists(text(), min_size=1, max_size=3))
    data = draw(lists(fixed_dictionaries({grouping: sampled_from(groups), measure:floats()}), max_size=20))
    return data, measure, grouping


class TestTwoLevelModel(TestCase):
    """Test case for a TwoLevelModel class"""


    def setUp(self):

        self.data = pd.DataFrame([{'grouping' : 'A', 'measure' : 2},
                                  {'grouping' : 'B', 'measure' : 3},
                                  {'grouping' : 'A', 'measure' : 5},
                                  {'grouping' : 'B', 'measure' : 5},
                                  {'grouping' : 'C', 'measure' : 2}])
        self.model = spig.TwoLevelModel(self.data, 'measure', 'grouping')


    @given(data_and_keys=generate_data_and_keys())
    def test_init(self, data_and_keys):
        data, measure, grouping = data_and_keys

        data_df = pd.DataFrame(data)

        model = spig.TwoLevelModel(data_df, measure, grouping)

        assert_frame_equal(model.data, data_df)
        self.assertEqual(model.measure, measure)
        self.assertEqual(model.grouping, grouping)



    def test_estimate_parameters(self):

        self.model.estimate_parameters()

        self.assertEqual(self.model.Y_stats.shape[0] + self.model.omitted_groups.shape[0],
                         self.data.groupby('grouping').ngroups)







