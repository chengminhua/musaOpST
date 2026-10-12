# SPDX-License-Identifier: MulanPSL-2.0
"""Independent reference consumes the same actual, quantized scalar input."""
import numpy as np


def scale(vector, scalar):
    return np.multiply(vector, scalar)
