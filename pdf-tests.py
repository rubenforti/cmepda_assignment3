
""" Unit tests for pdf.py module """

import unittest
# import numpy as np
from pdf import ProbabilityDensityFunction


class TestFunctions(unittest.TestCase):
    """
    Class for unit testing
    """

    def test_cdf(self):
        """ Unit testing of CDF """
        x_t1 = np.linspace(0, 1, 1000)
        y_t1 = 4*(x_t1**3)
        P = ProbabilityDensityFunction(x_t1, y_t1)
        y_test = x_t1**4
        y_cdf = P.cdf(x_t1)
        for i in range(0, 1000):
            self.assertAlmostEqual(y_test[i], y_cdf[i])

    def test_ppf(self):
        """ Unit testing of PPF """
        x_t2 = np.linspace(0, 1, 1000)
        y_t2 = 4*(x_t2**3)
        P = ProbabilityDensityFunction(x_t2, y_t2)
        y_test = x_t2**(1/4)
        y_ppf = P.ppf(x_t2)
        for i in range(0, 1000):
            self.assertAlmostEqual(y_ppf[i], y_test[i])


if __name__ == "__main__":
    unittest.main()
