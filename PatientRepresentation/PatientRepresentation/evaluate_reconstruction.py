import os

import dataset as dataset_m
from model import PatientModel
import patlearn_tools

import numpy as np
from sklearn import svm


def getDataset():
    base_dir = os.path.dirname(os.getcwd())
    base_dir = os.path.dirname(base_dir)

    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(data_dir, 'V7 Data')

    dataset = dataset_m.loadFromDir(data_dir, verbose=True)
    return dataset

def LeaveOneOutReconstruction(dataset):
    sum_err = 0.

    # model = PatientModel(max_iter=100)
    for patient_id in dataset.patients:
        for tissue_name in patient.tissues:
            removed_rep = dataset.removeValue(patient_id, tissue_name)

            model = PatientModel()
            #model.setWeightMult(patient_id, tissue_name, 0.)
            model.fit(dataset)

            predicted_rep = model.predict(patient_id, tissue_name)
            residual = removed_rep - predicted_rep

            sum_err += residual.dot(residual.T)[0,0]

            dataset.addValue(patient_id, tissue_name, removed_rep)

    return sum_err



if __name__ == '__main__':
    dataset = getDataset()
    for tissue in dataset.tissues.values():
        print '%s: %d %d' % (tissue.name, tissue.num_patients, len(tissue.patient_ids))
    for patient in dataset.patients.values():
        print '%s: %d %d' % (patient.id, patient.num_tissues, len(patient.tissue_names))

    model = PatientModel(max_iter=100)
    model.fit(dataset)

    tissue_name = dataset.tissues.keys()[0]
    tissue = dataset.tissues[tissue_name]

    # print tissue.value[tissue.rows[patient_id],:]
    # print model.predict(patient_id, tissue_name)

    total_var = 0.
    remaining_var = 0.
    for tissue_name in dataset.tissues:
        tissue = dataset.tissues[tissue_name]
        for patient_id in tissue.patient_ids:
            rep = dataset.getValue(patient_id, tissue_name)
            residual = model.predict(patient_id, tissue_name) - rep

            total_var += rep.T.dot(rep)[0,0]
            remaining_var += residual.T.dot(residual)[0,0]

    print 'total_var = {}'.format(total_var)
    print 'remaining_var = {}'.format(remaining_var)
    print 'LOO var = {}'.format(LeaveOneOutReconstruction(dataset))
    
    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(data_dir, 'V7 Covariates')

    filename = os.path.join(data_dir, 'GTEx_v7_Annotations_SubjectPhenotypesDS.txt')
    f = open(filename)
    f.readline()
    sexes = dict()
    train_sex_counts = dict()
    ages = dict()
    train_age_counts = dict()
    for line in f:
        items = line.strip().split()
        sex = int(items[1])
        sexes[items[0]] = sex
        if sex not in train_sex_counts:
            train_sex_counts[sex] = 0
        train_sex_counts[sex] += 1

        age = int(items[2].split('-')[0])
        ages[items[0]] = age
        if age not in train_age_counts:
            train_age_counts[age] = 0
        train_age_counts[age] += 1
    _, naive_sex = max([(train_sex_counts[sex], sex) for sex in train_sex_counts])
    _, naive_age = max([(train_age_counts[age], age) for age in train_age_counts])

    #print 'sex correlation: {}'.format(patlearn_tools.r2correlation(model, sexes, unbiased=True))
    #print 'random correlations:'
    #for _ in range(50):
    #    print patlearn_tools.randomCorrelation(model, unbiased=True)

    clf = svm.SVC()
    pids = []
    for id in sexes:
        if id in model.patients:
            pids += [id]
    pids = np.random.permutation(pids)
    ntotal = len(pids)
    ntrain = int(ntotal*4/5)
    clf.fit([model.patient_reps[id].tolist()[0] for id in pids[:ntrain]],
            [sexes[id] for id in pids[:ntrain]],
            [model.getWeight(id) for id in pids[:ntrain]])

    success = 0
    naive_success = 0
    for id in pids[ntrain:]:
        pred = clf.predict(model.patient_reps[id].tolist())
        if pred == sexes[id]:
            success += 1
        if naive_sex == sexes[id]:
            naive_success += 1

    print 'correctly predicted {}/{} sexes'.format(success, ntotal-ntrain)
    print 'naive: {}/{}'.format(naive_success, ntotal-ntrain)

    clf = svm.LinearSVC()
    pids = []
    for id in ages:
        if id in model.patients:
            pids += [id]
    ntotal = len(pids)
    ntrain = int(ntotal*4/5)
    clf.fit([model.patient_reps[id].tolist()[0] for id in pids[:ntrain]],
            [ages[id] for id in pids[:ntrain]],
            [model.getWeight(id) for id in pids[:ntrain]])
    
    success = 0
    naive_success = 0
    for id in pids[ntrain:]:
        pred = clf.predict(model.patient_reps[id].tolist())
        if pred == ages[id]:
            success += 1
        if naive_age == ages[id]:
            naive_success += 1

    print 'correctly predicted {}/{} ages'.format(success, ntotal-ntrain)
    print 'naive: {}/{}'.format(naive_success, ntotal-ntrain)




