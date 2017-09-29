"""Represents GTEx tissue data.

Attributes:
    tissues: {string --> Tissue}
    patients: {string --> Patient}
"""

import os

import tissue as tissue_m


class PatientTissueData(object):

    def __init__(self):
        self._tissues = dict()
        self._patients = dict()


    def addPatient(self, patient):
        self._patients[patient.id] = patient

    @property
    def patients(self):
        return self._patients

    @property
    def tissues(self):
        return self._tissues

def loadFromDir(directory_name, verbose=False):
    dataset = PatientTissueData()

    filenames = os.listdir(directory_name)

    for filename in filenames[:2]:
        if verbose:
            print 'reading from ' + filename
        tissue = tissue_m.loadFromFile(os.path.join(directory_name, filename), dataset, verbose=verbose)
        dataset.tissues[tissue.name] = tissue

    return dataset
    
