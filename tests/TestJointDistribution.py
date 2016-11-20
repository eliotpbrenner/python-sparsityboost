import unittest
from modelLearning.JointDistribution import JointDistribution
import numpy.testing as nptest
import numpy as np

class TestJointDistribution(unittest.TestCase):
    # test comment
    def test_set_joint_counts(self):
        distribution = JointDistribution(2,2)
        distribution.set_joint_counts(np.array([[3, 3],[3, 1]]))
        nptest.assert_array_equal(distribution.jointDistribution, np.array([[0.3, 0.3], [0.3, 0.1]]))
        nptest.assert_array_equal(distribution.A_counts, np.array([6,4]))
        nptest.assert_array_equal(distribution.B_counts, np.array([6,4]))
        nptest.assert_array_equal(distribution.A_distribution, np.array([0.6, 0.4]))
        nptest.assert_array_equal(distribution.B_distribution, np.array([0.6, 0.4]))
        nptest.assert_almost_equal(distribution.calculateMutualInformation(), 0.032189300825765904)

    def test_assymetric_joint_distribution(self):
        distribution = JointDistribution(2,2)
        distribution.set_joint_counts(np.array([[1, 5],[4, 0]]))
        nptest.assert_array_equal(distribution.jointDistribution, np.array([[0.1, 0.5], [0.4, 0.0]]))
        nptest.assert_array_equal(distribution.A_counts, np.array([6,4]))
        nptest.assert_array_equal(distribution.B_counts, np.array([5,5]))
        nptest.assert_array_equal(distribution.A_distribution, np.array([0.6,0.4]))
        nptest.assert_array_equal(distribution.B_distribution, np.array([0.5,0.5]))
        nptest.assert_array_equal(distribution.A_distribution, np.array([0.6, 0.4]))
        nptest.assert_almost_equal(distribution.calculateMutualInformation(), 0.42281045524016247)






if __name__ == '__main__':
    unittest.main()
