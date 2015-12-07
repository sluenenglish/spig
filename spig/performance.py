import pandas as pd


class TwoLevelModel(object):
    """Class for holding data and methods to compare performance

    Attributes
    ----------

    data : Pandas Dataframe
        Data to analyse and compare performance within
    first_level, second_level : string
        Strings containing column names of on which to construct
        groups in hierarchical models

    """

    def __init__(self, data, first_level, second_level):

        self.data = data

