"""Represents GTEx tissue data.

Attributes:
    tissues: {string --> Tissue}
    patients: {string --> Patient}
"""

import os
import copy

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
        return copy.copy(tissue.value[row:row+1, :])

    def removeValue(self, patient_id, tissue_name):
        pass

    def addValue(self, patient_id, tissue_name, value):
        pass

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

    total_var = 0.
    kept_var = 0.
    kept_dimensions = 0
    for filename in filenames:
        if verbose:
            print 'reading from ' + filename
        tissue, var_exp = tissue_m.loadFromFile(os.path.join(directory_name, filename), dataset, 
                                                verbose=verbose, explain_rat=10., ret_var=True)
        total_var += tissue.numPatients
        kept_var += var_exp * tissue.numPatients
        kept_dimensions += tissue.dimension
        dataset.tissues[tissue.name] = tissue

    if verbose:
        print 'total var explained: {}%'.format(100.*kept_var/total_var)
        print 'with total dimension: {}'.format(kept_dimensions)
        print 'in #tissues: {}'.format(len(dataset.tissues))

    return dataset
    
