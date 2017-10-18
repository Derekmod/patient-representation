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
        """Adds a Patient to database."""
        if patient.id in self._patients:
            print 'BUG: Rewriting patient'
        self._patients[patient.id] = patient

    def getValue(self, patient_id, tissue_name):
        """Get (full-ish) gene expression."""
        tissue = self.tissues[tissue_name]
        row = tissue.rows[patient_id]
        return tissue.value[row:row+1, :]

    @property
    def patients(self):
        return self._patients

    @property
    def tissues(self):
        return self._tissues

def loadFromDir(directory_name, verbose=False):
    """ Loads a dataset from a folder of gene expression files.
    Args:
        directory_name <string>: path to folder (absolute or relative)
        verbose <bool>: whether to output logging info
    Returns:
        a Dataset object with PCA applied.
    """
    dataset = PatientTissueData()

    filenames = os.listdir(directory_name)

    for filename in filenames:
        if verbose:
            print 'reading from ' + filename
        tissue = tissue_m.loadFromFile(os.path.join(directory_name, filename), dataset, verbose=verbose, explain_rat=10.)
        dataset.tissues[tissue.name] = tissue

    if verbose:
        total_var = 0.
        kept_var = 0.
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            for patient_id in tissue.patients:
                expr = dataset.getValue(patient_id, tissue_name)
                kept_var += expr.dot(expr.T)[0,0]
                total_var += 1.

        print 'total var explained: {}%'.format(100.*kept_var/total_var)

    return dataset
    
