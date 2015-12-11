from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


class TwoLevelModel(object):
    """Class for holding data and methods to compare performance

    Attributes
    ----------

    data : Pandas Dataframe
        Data to analyse and compare performance within
    measure : string
        String containing column name of performance indicator
    first_level, second_level : string
        Strings containing column names of on which to construct
        groups in hierarchical models

    """

    def __init__(self, data, measure, grouping):

        self.data = data
        self.measure = measure
        self.grouping = grouping


    def estimate_parameters(self):
        self.Y_stats = self.data.groupby(self.grouping).agg(['mean',
                                                             'count', 'sem'])[self.measure]
        self.Y_stats.rename(columns={'mean' : 'y_i',
                                     'count' : 'counts',
                                     'sem' : 'sigma'}, inplace=True)

        # omit groups with only one member
        omitted = self.Y_stats.counts > 1
        self.omitted_groups = self.Y_stats.index[~omitted]
        self.Y_stats = self.Y_stats[omitted]

        self.Y_var = self.data[self.measure].var()
        self.mu = self.data[self.measure].mean()

        self.Y_stats['tau2'] = self.Y_stats.apply(lambda s : self.Y_var - s.sigma ** 2, axis = 1)


        self.tau2 = self.Y_stats.tau2.mean()

        self.Y_stats['weight_i'] = self.Y_stats.apply(lambda s :
                                                      self.tau2 / (s.sigma ** 2 + self.tau2),
                                                      axis = 1)

        self.Y_stats['theta_i'] = self.Y_stats.apply(lambda s : s.weight_i * s.y_i + (1 - s.weight_i) * self.mu, axis = 1)


    def common_mean_comparison(self):

        try:
            self.Y_stats
        except:
            self.estimate_parameters()

        self.Y_stats['Z1'] = self.Y_stats.apply(lambda s : (s.y_i - self.mu) / s.sigma, axis = 1)
        self.Y_stats['P1'] = self.Y_stats.apply(lambda s : st.norm.cdf(s.Z1), axis = 1)


    def random_effects_comparison(self):

        try:
            self.Y_stats
        except:
            self.estimate_parameters()

        self.Y_stats['Z2'] = self.Y_stats.apply(lambda s :
                        np.sqrt(1 - s.weight_i) * ((self.mu - s.y_i) / s.sigma), axis = 1)
        self.Y_stats['P2'] = self.Y_stats.apply(lambda s : st.norm.cdf(s.Z2), axis = 1)


    def plot(self):

        lo = self.Y_stats.loc[self.Y_stats.idxmin()['y_i']]
        hi = self.Y_stats.loc[self.Y_stats.idxmax()['y_i']]

        lo_limit = lo.theta_i - 1.96 * lo.sigma
        hi_limit = hi.theta_i + 1.96 * hi.sigma

        x_axis = np.linspace(lo_limit, hi_limit, num=100)

        fig = plt.figure()
        sp = fig.add_subplot(111)

        sp.plot(x_axis, st.norm.pdf(x_axis, self.mu, self.tau2 ** 0.5), label=r'$\theta_i \mid \mu$')

        for i, row in self.Y_stats.iterrows():
            #print row['theta_i']
            sp.plot(x_axis, st.norm.pdf(x_axis, row.theta_i, row.sigma), label=i)

        sp.legend(loc='upper_left')

        return fig















