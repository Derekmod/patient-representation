"""Represents a tissue.

Attributes:
    value: 2d np.array. Rows are patients, columns are dimensions
    patients: [Patient] which patient is represented by given row
    rows: {string --> int} maps patient_id to row of its representation
    name: <string> name of tissue
    dimension: <int>
    #dataset: <Dataset> dataset that owns this Tissue
"""

import numpy as np
from sklearn.decomposition import PCA
import os

from patient import Patient

class Tissue(object):

    def __init__(self, name, dataset=None):
        self._name = name
        #self._dataset = dataset

        self._patients = dict()
        self._rows = dict()
        self._value = None

        self._patient_values = dict()

    def getValue(self, patient_id):
        if patient_id in self._patient_values:
            return None
        return self._patient_values[patient_id]

    def addValue(self, patient_id, val):
        self._patient_values[patient_id] = val

    def removeValue(self, patient_id):
        del self._patient_values[patient_id]

    @property
    def name(self):
        return self._name

    #@property
    #def dataset(self):
    #    return self._dataset

    @property
    def patients(self):
        return self._patients

    @property
    def rows(self):
        return self._rows

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        # TODO: arg check
        self._value = new_value

    @property
    def dimension(self):
        return self._value.shape[1]

    @property
    def numPatients(self):
        return len(self._patients)


def loadFromFile(filename, dataset, verbose=False, run_pca=True, explain_rat=4., ret_var=False):
    # TODO: arg check

    tissue_name = os.path.basename(filename).split('.')[0]
    tissue = Tissue(tissue_name, dataset)

    tissue_file = open(filename, 'r')

    patient_ids = tissue_file.readline().strip().split('\t')[4:]
    for patient_id in patient_ids:
        if patient_id not in dataset.patients:
            patient = Patient(patient_id)
            dataset.addPatient(patient)
        patient = dataset.patients[patient_id]
        patient.addTissue(tissue)
        
        tissue._rows[patient_id] = tissue.numPatients
        tissue._patients[patient_id] = patient

    # print 'got patients'

    raw_t = [[float(val_str)
              for val_str in line.strip().split('\t')[4:]]
             for line in tissue_file]

    # print 'got data'

    val = np.array(raw_t).T

    var_exp = 0.
    if run_pca:
        pca_model = PCA(n_components=50, copy=False)
        pca_model.fit_transform(val)
        #cov = val.T.dot(val)/(len(raw_t))

        #U, W, _ = np.linalg.svd(cov)

        #cum_var = np.cumsum(W**2)
        #cum_var = cum_var/cum_var[-1]
        cum_var = np.cumsum(pca_model.explained_variance_ratio_)
        explained_ratio = [float(cum_var[i])/float(i+1)
                           for i in range(len(cum_var))]
        
        best_dim = 0
        for dim in range(len(cum_var)):
            if explained_ratio[dim]*len(patient_ids) > explain_rat:
                best_dim = dim
        n_components = best_dim+1
        n_components = max(n_components, 8)

        #val = val.dot(U[:,:n_components])
        val = val[:,:n_components]
        var_exp = cum_var[n_components-1]
        
        if verbose:
            print tissue_name + ' has {} components to explain {}% variance for {} patients'.format(n_components, 100.*var_exp, len(patient_ids))

    elif verbose:
        print tissue_name + ' parsed'

    tissue._value = val
    for row, patient_id in enumerate(patient_ids):
        tissue.addValue(patient_id, val[row:row+1,:])

    if ret_var:
        return tissue, var_exp
    else:
        return tissue

