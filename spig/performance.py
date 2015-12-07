import pandas as pd


class Performance(object):
    """Class for holding data and methods to compare performance

    Attributes
    ----------

    data : Pandas Dataframe
        Data to analyse and compare performance within

    """

    def __init__(self, data):

        self.data = data
