"""Represents a tissue.

Attributes:
    value: 2d np.array. Rows are patients, columns are dimensions
    patients: [Patient] which patient is represented by given row
    rows: {string --> int} maps patient_id to row of its representation
    name: <string> name of tissue
    dimension: <int>
    #dataset: <Dataset> dataset that owns this Tissue
"""

from sklearn.decomposition import PCA
import numpy as np
from patient import Patient

class Tissue(object):

    def __init__(self, name):
        self._name = name

        self._patient_ids = set()
        self._dimension = 0

        self._patient_values = dict()

        self._technicals = dict()

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

    def runPCA(self, explain_rat=4., min_dim=8, verbose=False):
        val = np.concatenate([self.getValue(patient_id) for patient_id in self._patient_ids], axis=0)
        # TODO
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
            if explained_ratio[dim]*self.num_patients > explain_rat:
                best_dim = dim
        n_components = best_dim+1
        n_components = max(n_components, min_dim)

        #val = val.dot(U[:,:n_components])
        val = val[:,:n_components]
        var_exp = cum_var[n_components-1]
        
        if verbose:
            print tissue_name + ' has {} components to explain {}% variance for {} patients'.format(n_components, 100.*var_exp, len(patient_ids))
        pass

    def regressCovariates(self, cov_names=None):
        ''' Remove linear trends of certain covariates
        Args:
            cov_names [str]: names of covariates to regress upon (or None to regress on all)
        '''
        # TODO
        X_raw = []
        Y_raw = []

        for patient_id in self._patients:
            Y_raw += [self.getValue(patient_id)]
            X_raw += [self._technicals[(patient_id, tissue_name)]]

        Y = np.array([Y_raw]).T
        X = np.array(X_raw)

        self._technical_coefs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

        for patient_id, tissue_name in self.samples:
            reading = self.getValue(patient_id, tissue_name)
            
            pred = np.zeros(reading.shape)
            if (patient_id, tissue_name) in self._technicals:
                features = self._technicals[(patient_id, tissue_name)]
                pred = features.dot(self._technical_coefs)

            self.addValue(patient_id, tissue_name, reading-pred)
        pass

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