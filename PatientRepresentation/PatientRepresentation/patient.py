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
        self._tissues = dict()
        self._covariates = dict()

    def addTissue(self, tissue):
        self._tissues[tissue.name] = tissue

    def getTissueVal(self, tissue_name):
        """Gets 2d col vector rep. of patient's tissue sample."""
        tissue_rep, column = self._samples[tissue_name]
        return tissue_rep[:,column:column+1]

    def addCovariate(self, covariate_name, value):
        self._covariates[covariate_name] = value

    def getCovariate(self, covariate_name):
        if covariate_name not in self._covariates:
            return None
        return self._covariates[covariate_name]

    @property
    def id(self):
        return self._id

    def tissues(self):
        return self._tissues
