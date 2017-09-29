"""Represents a tissue.

Attributes:
    value: 2d np.array. Rows are patients, columns are dimensions
    patients: [Patient] which patient is represented by given row
    rows: {string --> int} maps patient_id to row of its representation
    name: <string> name of tissue
    dataset: <Dataset> dataset that owns this Tissue
"""

import numpy as np
# from sklearn.decomposition import PCA

from patient import Patient

class Tissue(object):

    def __init__(self, name, dataset):
        self._name = name
        self._dataset = dataset

        self._patients = []
        self._rows = dict()
        self._value = None

    @property
    def name(self):
        return self._name

    @property
    def dataset(self):
        return self._dataset

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
    def numPatients(self):
        return len(self._patients)


def loadFromFile(filename, dataset, verbose=False, run_pca=True, pca_var=0.9):
    # TODO: arg check

    tissue_name = filename.split('.')[0]
    tissue = Tissue(tissue_name, dataset)

    tissue_file = open(filename, 'r')

    patient_ids = tissue_file.readline().strip().split('\t')[4:]
    for patient_id in patient_ids:
        if patient_id not in tissue.dataset.patients:
            patient = Patient(patient_id)
            tissue.dataset.addPatient(patient)
        patient = tissue.dataset.patients[patient_id]
        patient.addTissue(tissue)
        
        tissue._rows[patient_id] = tissue.numPatients
        tissue._patients.append(patient)

    print 'got patients'

    raw_t = [[float(val_str)
              for val_str in line.strip().split('\t')[4:]]
             for line in tissue_file]

    print 'got data'

    val = np.array(raw_t).T

    if run_pca:
        #pca_model = PCA(n_components=50, copy=False)
        #pca_model.fit_transform(val)
        cov = val.T.dot(val)/(len(raw_t))

        U, W, _ = np.linalg.svd(cov)

        cum_var = np.cumsum(W**2)
        cum_var = cum_var/cum_var[-1]
        n_components = (cum_var<0.9).sum() + 1

        val = val.dot(U[:,:n_components])

        if verbose:
            print tissue_name + ' has {} components'.format(n_components)

        val = val[:n_components,:]
    elif verbose:
        print tissue_name + ' parsed'

    tissue._value = val

    return tissue

