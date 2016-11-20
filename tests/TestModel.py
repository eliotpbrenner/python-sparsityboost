import unittest
import os
from modelLearning.Model import Model
import numpy.testing as nptest
import numpy as np


class TestModel(unittest.TestCase):
    def setUp(self):
        self.dataFileDir = '/Users/eliotpbrenner/Projects/SparsityBoost/data/synthetic_examples/experiments/0'
        self.dataFileName = 'alarm1000.dat'
        self.dataFilePath = os.path.join(self.dataFileDir, self.dataFileName)
        self.model = Model()
        self.smallModel = Model()

    def testRead(self):
        self.model.readData(self.dataFilePath)
        self.assertEqual(self.model.data.sum().sum(), 18840)
        nptest.assert_array_equal(np.array(self.model.data.columns), np.arange(37))
        nptest.assert_array_equal(self.model.nval, np.array([2] * 37))

    def testFilterDataByDict(self):
        self.model.readData(self.dataFilePath)
        filtered_data = self.model.filterDataByDict({0: 0, 1: 1, 2: 0})
        self.assertEqual(filtered_data.sum().sum(), 2933)

    def testCalculateJointCounts(self):
        self.smallModel.readData(self.dataFilePath, nrows=10)
        nptest.assert_array_equal(self.smallModel.calculateJointCounts(0,1), np.array([[3, 3],[3, 1]]))
        nptest.assert_array_equal(self.smallModel.calculateJointCounts(1,2), np.array([[1, 5],[4, 0]]))

