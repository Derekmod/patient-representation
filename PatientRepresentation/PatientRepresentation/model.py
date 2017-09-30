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

    def __init__(self, dimension=8, max_iter=100):
        self._patient_reps = dict()
        self._tissue_centers = dict()
        self._tissue_transforms = dict()

        self._dimension = dimension

        self._max_iter = max_iter

    def fit(self, dataset):
        #self._dataset = dataset

        for patient_id in dataset.patients:
            self._patient_reps[patient_id] = np.random.randn(1, self.dimension)
            
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            self._tissue_centers[tissue_name] = np.zeros((1, tissue.dimension))

        for ep in range(self._max_iter):
            self.train_transforms(dataset)
            self.train_patients(dataset)
            self.train_centers(dataset)
            # self.normalize()

    def train_transforms(self, dataset):
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]

            residuals = tissue.value - self.tissue_centers[tissue_name]

            #patient_reps = concatenate vertically self.patient_reps[patient_id] for patient_id in tissue.patients
            patient_reps = np.concatenate([self.patient_reps[patient_id] 
                                           for patient_id in tissue.patients])

            # transform = least squares solver
            pinv = np.linalg.pinv(patient_reps)
            self.tissue_transforms[tissue_name] = pinv.dot(residuals)

    def train_patients(self, dataset):
        for patient_id in dataset.patients:
            patient = dataset.patients[patient_id]

            residuals = []
            transforms = []

            for tissue_name in patient.tissues:
                tissue = patient.tissues[tissue_name]

                patient_row = tissue.rows[patient_id]
                residuals.append(tissue.value[patient_row:patient_row+1,:] - self.tissue_centers[tissue_name])
                transforms.append(self.tissue_transforms[tissue_name])

            total_residual = np.concatenate(residuals, axis=1)
            total_transform = np.concatenate(transforms, axis=1)

            pinv = np.linalg.pinv(total_transform)
            self.patient_reps[patient_id] = total_residual.dot(pinv)

    def train_centers(self, dataset):
        for tissue_name in dataset.tissues:
            tissue = dataset.tissues[tissue_name]
            transform = self.tissue_transforms[tissue_name]

            sum_residual = -np.sum(tissue.value, axis=0).reshape(1, tissue.dimension)

            for patient_id in tissue.patients:
                sum_residual += self.patient_reps[patient_id].dot(transform)

            self.tissue_centers[tissue_name] = -sum_residual/tissue.numPatients

    def normalize(self):
        # TODO
        pass

    def predict(self, patient_id, tissue_name):
        return self.patient_reps[patient_id].dot(self.tissue_transforms[tissue_name]) + self.tissue_centers[tissue_name]


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