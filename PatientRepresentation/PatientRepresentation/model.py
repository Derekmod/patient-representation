"""Holds the parameters of the linear model.

Should have map of patient ids to patient indices.
Attributes:
    patient_reps: {string --> np.2darray} each rep is a H dimensional row vec
    tissue_centers: {string --> np.2darray} each center is a d dimensional row vec
    tissue_transforms: {string --> np.2darray} H x d matrix
    dimensions: <int> dimensionality of patient representations
"""

import numpy as np


class PatientModel(object):

    def __init__(self, dimension=8, max_iter=100, weight_inertia=5.):
        self._patient_reps = dict()
        self._tissue_centers = dict()
        self._tissue_transforms = dict()

        self._weight_mults = dict()

        self._dimension = dimension
        self._max_iter = max_iter
        self._weight_inertia = weight_inertia

        self._nsamples = dict()

    def fit(self, dataset):
        self._patient_samples = {id:len(dataset.patients[id].tissue_names) 
                                 for id in dataset.patients}
        self._tissue_samples = {name:len(dataset.tissues[name].patient_ids)
                                for name in dataset.tissues}
        self.getTissueValues(dataset)
        self.getPatientValues(dataset)

        for patient_id in dataset.patient_ids:
            self._patient_reps[patient_id] = np.random.randn(1, self.dimension)
            
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            self._tissue_centers[tissue_name] = np.zeros((1, tissue.dimension))
            
        prev_error = 1e16
        for ep in range(self._max_iter):
            #self.train_transforms(dataset)
            self.trainTransforms(dataset)
            self.trainPatients(dataset)
            #self.train_centers(dataset)
            error = self.errorFrac(dataset)
            print error
            if error > prev_error:
                break
                self.normalize()
                print 'normalizing'
            prev_error = error
            # self.normalize()

        self.normalize()

    def getTissueValues(self, dataset):
        self._tissue_values = dict()
        for tissue in dataset.tissues.values():
            value_list = [tissue.getValue(patient_id) for patient_id in tissue.patient_ids]
            #print value_list
            self._tissue_values[tissue.name] = np.concatenate(value_list)

    def getPatientValues(self, dataset):
        self._patient_values = dict()
        for patient in dataset.patients.values():
            self._patient_values[patient.id] = np.concatenate([patient.getValue(tissue_name)
                                                               for tissue_name in patient.tissue_names])


    def train_transforms(self, dataset):
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]

            residuals = self._tissue_values[tissue_name] - self.tissue_centers[tissue_name]

            #patient_reps = concatenate vertically self.patient_reps[patient_id] for patient_id in tissue.patients
            rep_list = [None]*tissue.numPatients
            for patient_id in tissue.patients:
                rep_list[tissue.rows[patient_id]] = self.patient_reps[patient_id]
            patient_reps = np.concatenate(rep_list)

            # transform = least squares solver
            pinv = np.linalg.pinv(patient_reps)
            #pinv = np.linalg.inv(patient_reps.T.dot(patient_reps)).dot(patient_reps.T)
            self.tissue_transforms[tissue_name] = pinv.dot(residuals)

    def trainTransforms(self, dataset):
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]

            #expr_list = [None]*tissue.numPatients
            #for patient_id in tissue.patients:
            #    expr = dataset.getValue(patient_id, tissue_name)
                #expr *= self.getSampleWeight(patient_id, tissue_name)
            #    expr_list[tissue.rows[patient_id]] = expr
            #expressions = np.concatenate(expr_list)
            expressions = self._tissue_values[tissue_name]
            
            #patient_reps = concatenate vertically self.patient_reps[patient_id] for patient_id in tissue.patients
            rep_list = [None]*tissue.numPatients
            for patient_id in tissue.patients:
                rep = self._patient_reps[patient_id]
                rep = np.concatenate([np.array([[1]]), rep], axis=1)
                rep *= self.getSampleWeight(patient_id, tissue_name)
                rep_list[tissue.rows[patient_id]] = rep
            pat_reps = np.concatenate(rep_list)

            extended_transform = np.linalg.pinv(pat_reps).dot(expressions)
            self.tissue_centers[tissue_name] = extended_transform[:1,:]
            self.tissue_transforms[tissue_name] = extended_transform[1:,:]


    def train_patients(self, dataset):
        for patient_id in dataset.patients:
            patient = dataset.patients[patient_id]

            residuals = []
            transforms = []

            for tissue_name in patient.tissues:
                residuals.append(patient.getValue(tissue_name) - self.tissue_centers[tissue_name])
                transforms.append(self.tissue_transforms[tissue_name])

            total_residual = np.concatenate(residuals, axis=1)
            total_transform = np.concatenate(transforms, axis=1)

            pinv = np.linalg.pinv(total_transform)
            self.patient_reps[patient_id] = total_residual.dot(pinv)

    def train_centers(self, dataset):
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            transform = self.tissue_transforms[tissue_name]

            sum_residual = -np.sum(self._tissue_values[tissue_name], axis=0).reshape(1, tissue.dimension)

            for patient_id in tissue.patients:
                sum_residual += self.patient_reps[patient_id].dot(transform)

            self.tissue_centers[tissue_name] = -sum_residual/tissue.numPatients

    def normalize(self):
        # normalize mean
        patient_mean = np.mean(self.patient_mat, axis=0)

        for tissue_name in self.tissues:
            self._tissue_centers[tissue_name] += patient_mean.dot(self._tissue_transforms[tissue_name])

        for patient_id in self.patients:
            self.patient_reps[patient_id] -= patient_mean

        # normalize variance
        reps = self.patient_mat
        patient_cov = reps.T.dot(reps)/self.num_patients
        U, W, V = np.linalg.svd(patient_cov)
        Einv = U.dot(np.diag(W**-.5))
        E = np.diag(W**.5).dot(V)

        for patient_id in self.patients:
            self.patient_reps[patient_id] = self.patient_reps[patient_id].dot(Einv)

        for tissue_name in self.tissues:
            self.tissue_transforms[tissue_name] = E.dot(self.tissue_transforms[tissue_name])

    def predict(self, patient_id, tissue_name):
        return self.patient_reps[patient_id].dot(self.tissue_transforms[tissue_name]) + self.tissue_centers[tissue_name]

    def errorFrac(self, dataset, weighted=True):
        total_var = 0.
        remaining_var = 0.
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            for patient_id in tissue.patients:
                rep = dataset.getValue(patient_id, tissue_name)
                residual = self.predict(patient_id, tissue_name) - rep
                weight = 1.
                if weighted:
                    weight = self.getSampleWeight(patient_id, tissue_name)

                total_var += rep.T.dot(rep)[0,0] * weight
                remaining_var += residual.T.dot(residual)[0,0] * weight

        return remaining_var/total_var

    def freeParameters(self):
        pat_params = self._dimension * self.num_patients
        pat_params -= self.dimension * (self.dimension+1) / 2

        # PENDING: add transform params based on dimension of tissue_space

        return pat_params


    @property
    def patient_reps(self):
        return self._patient_reps

    @property
    def tissue_centers(self):
        return self._tissue_centers

    @property
    def tissue_transforms(self):
        return self._tissue_transforms

    @property
    def dimension(self):
        return self._dimension

    @property
    def num_patients(self):
        return len(self._patient_reps)

    @property
    def patients(self):
        return self._patient_reps.keys()

    @property
    def tissues(self):
        return self._tissue_centers.keys()

    @property
    def patient_mat(self):
        return np.concatenate([self.patient_reps[id] for id in self.patients],
                              axis=0)

    @property
    def weights(self):
        # TODO: calculate real weights
        return {id: 1. for id in self.patients}

    def setWeightMult(self, patient_id, tissue_name, mult):
        self._weight_mults[(patient_id, tissue_name)] = mult

    def getWeight(self, patient_id):
        n_sample = self._nsamples[patient_id]
        return float(n_sample) / float(self._weight_inertia + n_sample)

    def getSampleWeight(self, patient_id, tissue_name):
        key = (patient_id, tissue_name)
        mult = 1.
        if key in self._weight_mults:
            mult = self._weight_mults[key]
        return self.getWeight(patient_id) * mult
