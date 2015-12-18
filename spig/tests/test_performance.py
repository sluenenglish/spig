from unittest import TestCase
import spig
from pandas.util.testing import assert_frame_equal
from hypothesis import given, example, assume
from hypothesis.strategies import text, floats, fixed_dictionaries, composite, lists, sampled_from, integers

import pandas as pd

#@composite
#def generate_data_and_keys(draw):
    #measure = draw(text(min_size=1))
    #grouping = draw(text(min_size=1))
    ##assume(measure != grouping)
    #groups = draw(lists(text(), min_size=1, max_size=3))
    #data = draw(lists(fixed_dictionaries({grouping: sampled_from(groups), measure:floats()}), max_size=20))
    #return data, measure, grouping
#

@composite
def generate_data_and_keys(draw, max_length=100, max_groups=20):
    measure_name = draw(text(min_size=1))
    groups_name = draw(text(min_size=1))
    assume(measure_name != groups_name)

    group_set = draw(lists(text(), min_size=1, max_size=max_groups))

    size = draw(integers(min_value=1, max_value = max_length))
    measures = draw(lists(floats(), min_size=size, max_size=size))
    groups = draw(lists(sampled_from(group_set),
                        min_size=size, max_size=size))
    data = [{measure_name: m, groups_name : g}
                for m,g in zip(measures, groups)]

    data_df = pd.DataFrame(data)

    return measure_name, groups_name, data_df

class TestUtilityFunctions(TestCase):
    """Tess for functions used in the tests"""
    @given(max_length=integers(min_value=1, max_value=200), max_groups=integers(min_value=1, max_value=40))
    def test_generate_data_and_keys(self, max_length, max_groups):
        output = generate_data_and_keys(max_length=max_length,
                                        max_groups=max_groups).example()
        # Assert correct types:
        self.assertEqual(len(output), 3)
        self.assertIsInstance(output, tuple)
        self.assertIsInstance(output[0], unicode)
        self.assertIsInstance(output[1], unicode)
        self.assertIsInstance(output[2], pd.DataFrame)

        # Assert constraints:
        self.assertNotEqual(output[0], output[1])
        for o in output[:-1]:
            self.assertGreater(len(o), 0)

        # Check data frame
        # 2 columns
        self.assertEqual(output[2].shape[1], 2)

        # At most max_length rows
        self.assertLessEqual(output[2].shape[0], max_length)

        # At most max_groups types of groups
        self.assertLessEqual(len(output[2][output[1]].unique()), max_length)

        # Checking types of each column:
        self.assertEqual(str(output[2][output[0]].dtypes), 'float64')
        self.assertEqual(str(output[2][output[1]].dtypes), 'object')


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
    @example(data_and_keys=('measure', 'grouping', None))
    def test_init(self, data_and_keys):

        measure_name, groups_name, data_df = data_and_keys

        if data_df is None:
            data_df = self.data

        model = spig.TwoLevelModel(data_df, measure_name, groups_name)

        assert_frame_equal(model.data, data_df)
        self.assertEqual(model.measure, measure_name)
        self.assertEqual(model.grouping, groups_name)


    @given(data_and_keys=generate_data_and_keys())
    @example(data_and_keys=('measure', 'grouping', None))
    def test_estimate_parameters(self, data_and_keys):


        measure_name, groups_name, data_df = data_and_keys

        if data_df is None:
            data_df = self.data

        model = spig.TwoLevelModel(data_df, measure_name, groups_name)

        model.estimate_parameters()

        self.assertEqual(model.Y_stats.shape[0] + model.omitted_groups.shape[0],
                         data_df.groupby(groups_name).ngroups)
