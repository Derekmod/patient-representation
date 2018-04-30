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

import collapse_logging as logging

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

    def runPCA(self, explain_rat=4., min_dim=8, verbose=False, var_per_dim=2., max_components=50):
        val = np.concatenate([self.getValue(patient_id) for patient_id in self._patient_ids], axis=0)
        initial_var = val.dot(val.T).trace()
        shape1 = val.shape
        # TODO
        pca_model = PCA(n_components=max_components, copy=False)
        val = pca_model.fit_transform(val)
        shape2 = val.shape
        #cov = val.T.dot(val)/(len(raw_t))

        #U, W, _ = np.linalg.svd(cov)

        #cum_var = np.cumsum(W**2)
        #cum_var = cum_var/cum_var[-1]
        cum_var = np.cumsum(pca_model.explained_variance_ratio_).tolist() + [np.sum(pca_model.explained_variance_ratio_)]
        #explained_ratio = [float(cum_var[i])/float(i+1)
        #                   for i in range(len(cum_var))]
        
        best_dim = 0
        for dim in range(len(cum_var)):
            var_exp = self.num_patients * pca_model.explained_variance_ratio_[dim]
            if var_exp >= var_per_dim:
                best_dim = dim
            else:
                break
        #n_components = best_dim+1
        n_components = best_dim
        n_components = max(n_components, min_dim)

        #val = val.dot(U[:,:n_components])
        val = val[:,:n_components]
        shape3 = val.shape
        final_var = val.dot(val.T).trace()
        var_exp = cum_var[n_components]

        for idx, patient_id in enumerate(self._patient_ids):
            self.addValue(patient_id, val[idx:idx+1,:])

        self._dimension = n_components
        
        if verbose:
            logging.log( self.name + ' has {} components to explain {}% variance for {} patients'.format(n_components, 100.*var_exp, len(self.patient_ids)))
            #logging.log('\t' + str(shape1) + '-->' + str(shape2) + '-->' + str(shape3))
            logging.log('\tvar: %f --> %f (%f fraction?)' % (initial_var, final_var, var_exp))

    def regressCovariates(self, cov_names=None):
        ''' Remove linear trends of certain covariates
        Args:
            cov_names [str]: names of covariates to regress upon (or None to regress on all)
        '''
        # TODO
        X_raw = []
        Y_raw = []

        technical_labels = set()
        for patient_id in self._patient_ids:
            sample_id = (patient_id, self.name)
            if sample_id in self._technicals:
                technical_labels |= set(self._technicals[sample_id].keys())
        technical_labels = list(technical_labels)
        logging.log('%d technicals for %s' % (len(technical_labels), self.name))

        for patient_id in self._patient_ids:
            Y_raw += [self.getValue(patient_id)]

            if patient_id in self._technicals:
                X_raw += [[]]

                tech = self._technicals[patient_id]
                for label in technical_labels:
                    if label in tech:
                        X_raw[-1] += [tech[label]]
                    else:
                        X_raw[-1] += [0.]
            else:
                X_raw += [[0.] * len(technical_labels)]
            #X_raw += [self._technicals[(patient_id, self.name)]]

        #Y = np.array([Y_raw]).T
        Y = np.concatenate(Y_raw, axis=0)
        X = np.array(X_raw)

        nsample, dim = X.shape
        #logging.log('X, Y shapes: ' + str(X.shape) + ',' + str(Y.shape))

        self._technical_coefs = np.linalg.inv(X.T.dot(X) + np.eye(dim)).dot(X.T).dot(Y)

        for idx, patient_id in enumerate(self._patient_ids):
            reading = self.getValue(patient_id)
            
            pred = X[idx:idx+1].dot(self._technical_coefs)

            self.addValue(patient_id, reading-pred)
        pass

    def normalize(self, verbose=False):
        val = np.concatenate([self.getValue(patient_id) for patient_id in self._patient_ids], axis=0)
        initial_m2 = np.trace(val.dot(val.T))
        val -= np.mean(val, axis=0)
        var = np.trace(val.dot(val.T))/len(self._patient_ids)
        inv = var ** -0.5
        val = val * inv
        final_m2 = np.trace(val.dot(val.T))
        for idx, patient_id in enumerate(self._patient_ids):
            self.addValue(patient_id, val[idx:idx+1])

        if verbose:
            logging.log('\treported var: %f --> %f with %d patients' % (initial_m2, final_m2, self.num_patients))
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