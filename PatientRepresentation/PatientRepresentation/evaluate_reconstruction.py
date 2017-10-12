import os

import dataset as dataset_m
from model import PatientModel
import patlearn_tools


if __name__ == '__main__':
    base_dir = os.path.dirname(os.getcwd())
    base_dir = os.path.dirname(base_dir)

    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(data_dir, 'V7 Data')

    dataset = dataset_m.loadFromDir(data_dir, verbose=True)

    model = PatientModel(max_iter=400)
    model.fit(dataset)

    tissue_name = dataset.tissues.keys()[0]
    tissue = dataset.tissues[tissue_name]

    patient_id = tissue.patients.keys()[0]
    patient = tissue.patients[patient_id]

    # print tissue.value[tissue.rows[patient_id],:]
    # print model.predict(patient_id, tissue_name)

    total_var = 0.
    remaining_var = 0.
    for tissue_name in dataset.tissues:
        tissue = dataset.tissues[tissue_name]
        for patient_id in tissue.patients:
            rep = dataset.getValue(patient_id, tissue_name)
            residual = model.predict(patient_id, tissue_name) - rep

            total_var += rep.T.dot(rep)[0,0]
            remaining_var += residual.T.dot(residual)[0,0]

    print 'total_var = {}'.format(total_var)
    print 'remaining_var = {}'.format(remaining_var)
    
    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(data_dir, 'V7 Covariates')

    filename = os.path.join(data_dir, 'GTEx_v7_Annotations_SubjectPhenotypesDS.txt')
    f = open(filename)
    f.readline()
    sexes = dict()
    for line in f:
        items = line.strip().split()
        sexes[items[0]] = int(items[1])

    print 'sex correlation: {}'.format(patlearn_tools.r2correlation(model, sexes))
    print 'random correlations:'
    for _ in range(50):
        print patlearn_tools.randomCorrelation(model)


