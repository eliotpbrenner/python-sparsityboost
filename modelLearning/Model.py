import pandas as pd
import numpy as np


class Model(object):
    def __init__(self):
        nodes = 0
        nval = np.zeros(1)
        data = pd.DataFrame()

    def readData(self, dataFile, nrows = 1e6):
        """
        :param dataFile: space separated, integer valued, nothing but data (no header)
        :param nrows: only read this many rows, default 1e6.
        :return:
        """
        self.data = pd.read_csv(dataFile, sep=" ", header=None, nrows=nrows)
        self.nval = np.array(self.data.max())
        self.nval = self.nval + np.ones(np.shape(self.nval))

    def filterDataByDict(self, data_filter):
        """
        :param data_filter: Dictionary of the form {...,A: v,...} where A
         is the (index of) a variable of the data and v is in Val(A)
        :return: the subset of datapoints d for which d[A]=v.
        """
        # http://stackoverflow.com/questions/38137821/filter-dataframe-using-dictionary
        data_filter_list_values = {k: [v] for k, v in data_filter.items()}
        return self.data[self.data.isin(data_filter_list_values).sum(1) == len(data_filter_list_values.keys())]

    def calculateJointCounts(self, A, B):
        """
        :param A: index of the first variable in pair
        :param B: index of the second variable in pair
        :return: Joint, unconditioned, unnormalized empirical counts of variable pair (A,B)
        """
        A_nval = int(self.nval[A])
        B_nval = int(self.nval[B])
        jointCounts = np.zeros((A_nval, B_nval))
        for u in range(A_nval):
            for v in range(B_nval):
                jointCounts[u, v] = len(self.filterDataByDict({A: u, B: v}))
        return jointCounts

    def calculateJointCountsAndMarginalCounts(self, A, B):
        """
        :param A:
        :param B:
        :return:
        """


