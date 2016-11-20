import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointDistribution(object):
    def __init__(self, A_nval, B_nval):
        self.jointCounts = np.zeros((A_nval, B_nval))
        self.jointDistribution = np.zeros((A_nval, B_nval))
        self.A_counts = np.zeros(A_nval)
        self.B_counts = np.zeros(B_nval)
        self.A_distribution = np.zeros(A_nval)
        self.B_distribution = np.zeros(B_nval)

    def _calculate_joint_distribution(self):
        normalizingFactor = self.jointCounts.sum()
        if np.abs(normalizingFactor) < 1e-30:
            logger.warn('Cannot normalize distribution because all entries 0')
        else:
            self.jointDistribution = self.jointCounts / normalizingFactor

    def _calculate_marginal_counts(self):
        self.A_counts = self.jointCounts.sum(axis=1)
        self.B_counts = self.jointCounts.sum(axis=0)

    def _calculate_marginal_distributions(self):
        normalizing_factor = self.jointCounts.sum()
        if np.abs(normalizing_factor) < 1e-30:
            logger.warn('Cannot normalize distribution because all entries 0')
        else:
            self.A_distribution = self.A_counts / normalizing_factor
            self.B_distribution = self.B_counts / normalizing_factor

    def set_joint_counts(self, joint_counts):
        assert np.shape(joint_counts) == np.shape(self.jointCounts)
        self.jointCounts = joint_counts
        self._calculate_joint_distribution()
        self._calculate_marginal_counts()
        self._calculate_marginal_distributions()

    def calculateMutualInformation(self):
        mutual_information = 0
        for u in range(np.shape(self.jointDistribution)[0]):
            for v in range(np.shape(self.jointDistribution)[1]):
                if self.jointDistribution[u,v] < 1e-30:
                    continue
                mutual_information += self.jointDistribution[u, v] * np.log(
                    self.jointDistribution[u, v] / (self.A_distribution[u] * self.B_distribution[v]))
        return mutual_information
