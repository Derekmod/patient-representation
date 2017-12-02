"""Represents a tissue.

Attributes:
    value: 2d np.array. Rows are patients, columns are dimensions
    patients: [Patient] which patient is represented by given row
    rows: {string --> int} maps patient_id to row of its representation
    name: <string> name of tissue
    dimension: <int>
    #dataset: <Dataset> dataset that owns this Tissue
"""

from patient import Patient

class Tissue(object):

    def __init__(self, name):
        self._name = name

        self._patient_ids = set()
        self._dimension = 0

        self._patient_values = dict()

    def getValue(self, patient_id):
        if patient_id not in self._patient_ids:
            return None
        return self._patient_values[patient_id]

    def addValue(self, patient_id, val):
        self._patient_values[patient_id] = val
        self._patient_ids.add(patient_id)

    def removeValue(self, patient_id):
        value = self._patient_values[patient_id]
        del self._patient_values[patient_id]
        self._patient_ids.remove(patient_id)
        return value

    @property
    def name(self):
        return self._name

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def dimension(self):
        return self._dimension

    @property
    def num_patients(self):
        return len(self._patient_ids)