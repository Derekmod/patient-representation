"""Represents GTEx tissue data.

Attributes:
    tissues: {string --> Tissue}
    patients: {string --> Patient}
"""

import os
import copy
import pickle
import collapse_logging as logging

import numpy as np
import random

from tissue import Tissue
from patient import Patient
import attribute_parser as parser


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

    def addTechnicalsFromFile(self, filename, regress=False, verbose=False):
        ''' stores data in Tissue object'''
        labels = set()

        for sample_id, label, val in parser.regressableSampleAttributes(filename):
            patient_id, tissue_name = sample_id
            tissue = self.tissues[tissue_name]
            if patient_id not in tissue._technicals:
                tissue._technicals[patient_id] = dict()
            tissue._technicals[patient_id][label] = val
            labels.add(label)

        if regress:
            self.regressCovariates(cov_names=self._technical_labels, verbose=verbose)

        return labels

    def addCovariatesFromFile(self, filename, verbose=False):
        labels = set()
        for patient_id, label, val, in parser.regressableSubjectAttributes(filename):
            for tissue in self.tissues.values():
                if patient_id not in tissue._technicals:
                    tissue._technicals[patient_id] = dict()
                tissue._technicals[patient_id][label] = val
            labels.add(label)

        return labels

    def runPCA(self, tissue_name=None, var_per_dim=10., verbose=False):
        if tissue_name is None:
            if verbose:
                progress = logging.addNode('Running PCA', count=len(self._tissues))
            for tn in self._tissues:
                self._tissues[tn].runPCA(verbose=verbose)
                if verbose:
                    progress.step()

            if verbose:
                logging.closeNode()
        else:
            self._tissues[tissue_name].runPCA(var_per_dim=var_per_dim)

        for tissue_name in self.tissues:
            for patient_id in self.tissues[tissue_name].patient_ids:
                self.addValue(patient_id, tissue_name, self._tissues[tissue_name].getValue(patient_id))

    def regressCovariates(self, tissue_name=None, cov_names=None, verbose=True):
        ''' Remove linear trends of certain covariates.
        Args:
            cov_names [str]: names of covariates to regress upon (or None to regress on all)
        '''
        if tissue_name is None:
            if verbose:
                progress = logging.addNode('Regressing Covariates', count=len(self._tissues))
            for tn in self._tissues:
                if verbose:
                    #statement = logging.log('regressing on tissue: ' + tn)
                    pass
                self._tissues[tn].regressCovariates(self._technicals)
                if verbose:
                    progress.step()
            if verbose:
                logging.closeNode()

        else:
            self._tissues[tissue_name].regressCovariates(cov_names=cov_names)

        for tn in self.tissues:
            for patient_id in self.tissues[tn].patient_ids:
                self.addValue(patient_id, tn, self._tissues[tn].getValue(patient_id))

    def normalize(self, verbose=True):
        if verbose:
            progress = logging.addNode('Normalizing', count=len(self._tissues))
        for tissue in self._tissues.values():
            if verbose:
                logging.log('normalizing ' + tissue.name)
            tissue.normalize(verbose=verbose)
            if verbose:
                progress.step()
        if verbose:
            logging.closeNode()


    def split(self, seed=None, train_frac=0.8, val_frac=0.):
        ''' Split dataset into train, validate, test.
        Args:
            seed: random seed
        Returns:
            training sample ids <(patient_id, tissue_name)>
            test sample ids
        '''
        samples = []
        for sample in self.samples:
            samples += [sample]
        np.random.seed(seed)
        samples = np.random.permutation(samples)

        ntrain = int(len(samples) * train_frac)
        nvalid = int(len(samples) * (train_frac + val_frac))
        return samples[:ntrain], samples[ntrain:nvalid], samples[nvalid:]
        
    @property
    def patients(self):
        return self._patients

    @property
    def tissues(self):
        return self._tissues

    @property 
    def samples(self):
        ret = []
        for patient_id in self._patients:
            for tissue_name in self._patients[patient_id].tissue_names:
                ret += [(patient_id, tissue_name)]
        return ret

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

    @property
    def dimension(self):
        ans = 0
        for tissue in self.tissues.values():
            ans += tissue.dimension
        return ans

    @property
    def sum_dimension(self):
        ans = 0
        for tissue in self.tissues.values():
            ans += tissue.dimension * tissue.num_patients
        return ans

    @property
    def summary(self):
        ans = ''
        ans += '#tissues: %d, ' % len(self.tissues)
        ans += '#patients: %d, ' % len(self.patients)
        ans += '#samples: %d, ' % self.total_samples
        ans += 'dimension: %d, ' % self.dimension
        ans += 'sum_dimension: %d, ' % self.sum_dimension
        ans += 'variance: %f ' % self.total_variance
        return ans

def loadFromDir(directory_name, verbose=False):
    """ Loads a dataset from a folder of gene expression files.
    Args:
        directory_name <string>: path to folder (absolute or relative)
        verbose <bool>: whether to output logging info
    Returns:
    """
    dataset = PatientTissueData()

    filenames = os.listdir(directory_name)
    #load_progress = logging.logProgress('LOADING FILES', len(filenames))
    if verbose:
        load_progress = logging.addNode('LOADING FILES', count=len(filenames))
    for filename in filenames:
        if verbose:
            logging.log('reading from ' + filename)
        loadFromFile(os.path.join(directory_name, filename), dataset, 
                               verbose=verbose)
        load_progress.step()
        #break

    if verbose:
        logging.closeNode()

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

    for row, patient_id in enumerate(patient_ids):
        dataset.addValue(patient_id, tissue_name, val[row:row+1,:])

    dataset.tissues[tissue_name]._dimension = val.shape[1]

    return tissue_name

def loadFromPickle(filename):
    return pickle.load(open(filename, 'rb'))

