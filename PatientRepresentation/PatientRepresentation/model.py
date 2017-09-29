"""Holds the parameters of the linear model.

Should have map of patient ids to patient indices."""

import numpy as np


class PatientModel(object):

    def __init__(self):
        self.reps = np.array([[0]])