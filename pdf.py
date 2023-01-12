# Copyright (C) 2022 r.forti1@studenti.unipi.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Third assignment for the CMEPDA course, 2022/23."""

# --- Goal
# Create a ProbabilityDensityFunction class that is capable of throwing
# preudo-random number with an arbitrary distribution.

# (In practice, start with something easy, like a triangular distribution---the
# initial debug will be easier if you know exactly what to expect.)


# --- Specifications
# - the signature of the constructor should be __init__(self, x, y), where
#   x and y are two numpy arrays sampling the pdf on a grid of values, that
#   you will use to build a spline
# - [optional] add more arguments to the constructor to control the creation
#   of the spline (e.g., its order)
# - the class should be able to evaluate itself on a generic point or array of
#   points
# - the class should be able to calculate the probability for the random
#   variable to be included in a generic interval
# - the class should be able to throw random numbers according to the
#   distribution that it represents
# - [optional] how many random numbers do you have to throw to hit the
#   numerical inaccuracy of your generator?

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

    """
    Class describing a probability density function.
    -------------
    Parameters:
      x : array-like
          The array of x values to be passed to the pdf.
      y : array-like
          The array of y values to be passed to the pdf.
    """

    def __init__(self, x, y, k=3):
        """Constructor.
        """

        # Normalize the pdf, if it is not.
        norm = InterpolatedUnivariateSpline(x, y, k=k).integral(x[0], x[-1])
        y /= norm

        super().__init__(x, y, k=k)  # Eredita il costruttore dalla classe precedente

        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf, k=k)

        # Need to make sure that the vector I am passing to the ppf spline as
        # the x values has no duplicates --- and need to filter the y
        # accordingly.
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]

        self.ppf = InterpolatedUnivariateSpline(xppf, yppf, k=k)  # Cumulative
        # density function inverted --- with "x" in "y" place and vice-versa

    def prob(self, x_1, x_2):
        """
        Return the probability for the random variable to be included
        between xA and xB.

        Parameters
        ----------
        x_1: float or array-like
            The left bound for the integration.
        x_2: float or array-like
            The right bound for the integration.
        """
        return self.cdf(x_2) - self.cdf(x_1)

    def rnd(self, size=1000):
        """Return an array of random values from the pdf.
        Parameters
        ----------
        size: int
            The number of random numbers to extract.
        """
        return self.ppf(np.random.uniform(size=size))


if __name__ == "__main__":
    x_user = np.linspace(0, 1, 1000)
    y_user = x_user**3
    pdf = ProbabilityDensityFunction(x_user, y_user, k=3)
    yy = pdf.rnd(100000)
    plt.plot(x_user, pdf.ppf(x_user))
    plt.plot(x_user, pdf.cdf(x_user))
    #plt.plot(x, pdf.ppf(x)+pdf.cdf(x))
    plt.show()
