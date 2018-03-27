"""Represents GTEx tissue data.

Attributes:
    tissues: {string --> Tissue}
    patients: {string --> Patient}
"""

import os
import copy
import pickle

import numpy as np

from tissue import Tissue
from patient import Patient


class PatientTissueData(object):
    def __init__(self):
        self._tissues = dict()
        self._patients = dict()
        self._technicals = dict()  # technicals[patient_id][tissue_name] = {label --> val}
        self._technical_labels = []

    def addPatient(self, patient):
        """Adds a Patient to database."""
        if patient.id in self._patients:
            print 'BUG: Rewriting patient'
        self._patients[patient.id] = patient

    def getValue(self, patient_id, tissue_name):
        """Get (full-ish) gene expression."""
        return self._tissues[tissue_name].getValue(patient_id)

    def removeValue(self, patient_id, tissue_name):
        val = self.getValue(patient_id, tissue_name)
        self._patients[patient_id].removeValue(tissue_name)
        if self._patients[patient_id].num_tissues == 0:
            del self._patients[patient_id]
        self._tissues[tissue_name].removeValue(patient_id)
        if self._tissues[tissue_name].num_patients == 0:
            del self._tissues[tissue_name]
        # TODO: check if patient or tissue has no data
        return val

    def addValue(self, patient_id, tissue_name, value):
        if patient_id not in self._patients:
            self._patients[patient_id] = Patient(patient_id)
        if tissue_name not in self._tissues:
            self._tissues[tissue_name] = Tissue(tissue_name)

        self._patients[patient_id].addValue(tissue_name, value)
        self._tissues[tissue_name].addValue(patient_id, value)

    def pickle(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def addTechnicalsFromFile(self, filename, regress=True):
        ''' stores data in Tissue object'''
        f = open(filename)
        labels = f.readline().strip().split()

        for line in f:
            items = line.strip().split()

            id = items[0]
            id_components = items.split('-')
            patient_id = '-'.join(id_components[:2])
            tissue_name = items[10]
            tissue = self._tissues[tissue_name]
            
            #if patient_id not in self._technicals:
            #    self._technicals[patient_id] = dict()

            #if tissue_name not in self._technicals[patient_id]:
            #    self._technicals[(patient_id, tissue_name)] = dict()

            for i in range(1, len(items)):
                try:
                    val = float(items[i])
                    #technicals[(patient_id, tissue_name)][labels[i]] = val
                    tissue._technicals[labels[i]][patient_id] = val
                    if labels[i] not in self._technical_labels:
                        self._technical_labels += [labels[i]]
                except:
                    pass

        for sample_id in self.samples:
            self._technicals[sample_id] = np.array([technicals[sample_id][label]
                                                    for label in self._technical_labels])

    def runPCA(self, tissue_name=None):
        if tissue_name is None:
            for tn in self._tissues:
                self._tissues[tn].runPCA()
        else:
            self._tissues[tissue_name].runPCA()

    def regressCovariates(self, tissue_name=None, cov_names=None):
        ''' Remove linear trends of certain covariates.
        Args:
            cov_names [str]: names of covariates to regress upon (or None to regress on all)
        '''
        if tissue_name is None:
            for tn in self._tissues:
                self._tissues[tn].regressCovariates(cov_names)
        else:
            self._tissues[tissue_name].regressCovariates(cov_names)

    def split(self, seed=None):
        ''' Split dataset into train, validate, test.
        Args:
            seed: random seed
        Returns:
            ?
        '''
        pass
        
    @property
    def patients(self):
        return self._patients

    @property
    def tissues(self):
        return self._tissues

    @property 
    def samples(self):
        for patient_id in self._patients:
            for tissue_name in self._patients[patient_id].tissues:
                yield (patient_id, tissue_name)

    @property
    def total_samples(self):
        ans = 0
        for tissue in self.tissues.values():
            ans += tissue.num_patients
        return ans

    @property
    def total_variance(self):
        ans = 0.
        for tissue in self.tissues.values():
            for patient_id in tissue.patient_ids:
                rep = self.getValue(patient_id, tissue.name)
                ans += rep.dot(rep.T)[0,0]
        return ans

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
        kept_var = loadFromFile(os.path.join(directory_name, filename), dataset, 
                               verbose=verbose, explain_rat=10., ret_var=True)

    for tissue in dataset.tissues.values():
        #kept_var += var_exp * tissue.num_patients
        kept_dimensions += tissue.dimension
        total_var += tissue.num_patients

    if verbose:
        print 'total var explained: {}%'.format(100.*kept_var/total_var)
        print 'with total dimension: {}'.format(kept_dimensions)
        print 'in #tissues: {}'.format(len(dataset.tissues))

    return dataset
    

def loadFromFile(filename, dataset, verbose=False, run_pca=False):
    # FUTURE: arg check

    tissue_name = os.path.basename(filename).split('.')[0]
    tissue_file = open(filename, 'r')

    patient_ids = tissue_file.readline().strip().split('\t')[4:]

    raw_t = [[float(val_str)
              for val_str in line.strip().split('\t')[4:]]
             for line in tissue_file]

    val = np.array(raw_t).T

    if verbose:
        print tissue_name + ' parsed'

    for row, patient_id in enumerate(patient_ids):
        dataset.addValue(patient_id, tissue_name, val[row:row+1,:])

    dataset.tissues[tissue_name]._dimension = val.shape[1]

    return var_exp * dataset.tissues[tissue_name].num_patients

def loadFromPickle(filename):
    return pickle.load(open(filename, 'rb'))

