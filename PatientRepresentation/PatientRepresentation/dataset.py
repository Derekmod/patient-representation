"""Represents GTEx tissue data.

Attributes:
    tissues: {string --> Tissue}
    patients: {string --> Patient}
"""

import os
import copy

from tissue import Tissue
from patient import Patient


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
        val = self.getValue(patient_id, tissue_name)
        self._patients[patient_id].removeValue(tissue_name)
        self._tissues[tissue_name].removeValue(patient_id)
        # TODO: check if patient or tissue has no data
        return val

    def addValue(self, patient_id, tissue_name, value):
        if patient_id not in self._patients:
            self._patients[patient_id] = Patient(patient_id)
        if tissue_name not in self._tissues:
            self._tissues[tissue_name] = Tissue(tissue_name)

        self._patients[patient_id].addValue(tissue_name, value)
        self._tissues[tissue_name].addValue(patient_id, value)

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
        tissue, var_exp = loadFromFile(os.path.join(directory_name, filename), dataset, 
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
    

def loadFromFile(filename, dataset, verbose=False, run_pca=True, explain_rat=4., ret_var=False):
    # TODO: arg check

    tissue_name = os.path.basename(filename).split('.')[0]
    tissue = Tissue(tissue_name)

    tissue_file = open(filename, 'r')

    patient_ids = tissue_file.readline().strip().split('\t')[4:]

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
        
    #for patient_id in patient_ids:
    #    if patient_id not in dataset.patients:
    #        patient = Patient(patient_id)
    #        dataset.addPatient(patient)

    tissue._value = val
    for row, patient_id in enumerate(patient_ids):
        dataset.addValue(patient_id, tissue_name, val[row:row+1,:])
        #tissue.addValue(patient_id, val[row:row+1,:])
        #dataset.patients[patient_id].addValue(tissue_name, val[row:row+1,:])
        

    if ret_var:
        return tissue, var_exp
    else:
        return tissue

