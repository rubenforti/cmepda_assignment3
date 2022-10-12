#!/usr/bin/env python
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
# - the class should be able to throw random numbers according to the distribution
#   that it represents
# - [optional] how many random numbers do you have to throw to hit the
#   numerical inaccuracy of your generator?

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class ProbabilityDensityFunction:

    """
    Class describing a probability density function.
    -------------
    Parameters:
      x : array-like
          The array of x values to be passed to the pdf.
      y : array-like
          The array of y values to be passed to the pdf.
    """

    def __init__(self, x, y):
        """Constructor"""
        curve = InterpolatedUnivariateSpline(x, y, k=3)


if __name__ == "__main__":
    x = np.linspace(0, 2, 10)
    y = 1-0.5*x
