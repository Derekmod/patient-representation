"""Represents one patient.

Attributes:
    id: <string> id specifying patient in GTEx
Private Attributes:
    _samples: <string --> Tissue>
    _covariates: <string --> number>
"""

class Patient(object):

    def __init__(self, id):
        self._id = id
        self._tissue_names = set()
        self._covariates = dict()
        self._values = dict()

    def getValue(self, tissue_name):
        if tissue_name not in self._tissue_names:
            return None
        return self._values[tissue_name]

    def removeValue(self, tissue_name):
        value = self._values[tissue_name]
        del self._values[tissue_name]
        self._tissue_names.remove(tissue_name)
        return value

    def addValue(self, tissue_name, value):
        self._values[tissue_name] = value
        self._tissue_names.add(tissue_name)

    def addCovariate(self, covariate_name, value):
        self._covariates[covariate_name] = value

    def getCovariate(self, covariate_name):
        if covariate_name not in self._covariates:
            return None
        return self._covariates[covariate_name]

    @property
    def id(self):
        return self._id

    @property
    def tissue_names(self):
        return self._tissue_names

    @property
    def num_tissues(self):
        return len(self._tissue_names)
