import os
import sys
import collapse_logging as logging

import dataset as dataset_m
from model import PatientModel
import patlearn_tools

import numpy as np
from sklearn import svm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("--data-dir", type=str)
parser.add_argument("--pickle-dir", type=str)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("--output-dir", type=str)
parser.add_argument("--dimension", type=int)
parser.add_argument("--inertia", type=float)
parser.add_argument("--max-iter", type=int)
parser.add_argument("--technicals", type=str)
parser.add_argument("--covariates", type=str)
#parser.add_argument("--load", type=str, help="file to load weights from")
#parser.add_argument("--save", help="file to save weights to", type=str)
#parser.add_argument("--nepochs", type=int, help="max # of epochs for training")
#parser.add_argument("--data", type=str, help="file to load dataset from")
args = parser.parse_args()
if args.dimension is None:
    args.dimension = 5
if args.inertia is None:
    args.inertia = 10.
if args.max_iter is None:
    args.max_iter = 50
if args.pickle_dir is None:
    args.pickle_dir = './pickled/'
if args.data_dir:
    args.data_dir = args.data_dir.strip()
if args.technicals:
    args.technicals = args.technicals.strip()
if args.covariates:
    args.covariates = args.covariates.strip()


def getDataset():
    base_dir = os.path.dirname(os.getcwd())
    base_dir = os.path.dirname(base_dir)

    data_dir = args.data_dir
    if not args.data_dir:
        data_dir = os.path.join(base_dir, 'data')
        data_dir = os.path.join(data_dir, 'V7 Data')

    if not os.path.isdir(args.pickle_dir):
        os.makedirs(args.pickle_dir)

    pickle_filename = os.path.join(args.pickle_dir, 'dataset.pickle')
    if os.path.exists(pickle_filename):
        return dataset_m.loadFromPickle(pickle_filename)
    else:
        dataset = dataset_m.loadFromDir(data_dir, verbose=True)
        logging.log('adding technicals')
        dataset.addTechnicalsFromFile(args.technicals, regress=True)
        dataset.runPCA()
        dataset.pickle(pickle_filename)
        return dataset

def LeaveOneOutReconstruction(dataset, max_iter=50, dimension=5, tissue_inertia=10., patient_inertia=10., logstream=None, min_samples=2):
    avg_err = dataset.total_variance / dataset.total_samples

    samples = []
    for tissue in dataset.tissues.values():
        for patient_id in tissue.patient_ids:
            samples += [(tissue.name, patient_id)]
    samples = np.random.permutation(samples)
    progress = logging.logProgress('LOOR progress', len(samples))
    print 'will need to go through %d tests:' % len(samples)

    # model = PatientModel(max_iter=100)
    sum_err = 0.
    sum_err2 = 0.
    sum_var = 0.

    weighted_sum_err = 0.
    weighted_sum_err2 = 0.
    sum_weight = 0.
    max_weight = 0.
    for sample_no, (tissue_name, patient_id) in enumerate(samples):
        if dataset.patients[patient_id].num_tissues < min_samples:
            continue
        if dataset.tissues[tissue_name].num_patients < min_samples:
            continue
        removed_rep = dataset.removeValue(patient_id, tissue_name)
        rep_var = removed_rep.dot(removed_rep.T)[0,0]
        sum_var += rep_var

        model = PatientModel(max_iter=max_iter, dimension=dimension, weight_inertia=tissue_inertia)
        #model.setWeightMult(patient_id, tissue_name, 0.)
        model.fit(dataset, verbose=False)

        predicted_rep = model.predict(patient_id, tissue_name)
        residual = removed_rep
        if predicted_rep is not None:
            residual = removed_rep - predicted_rep

        err = residual.dot(residual.T)[0,0]
        if logstream is not None:
            # scale, err, #patients, #tissues, patient_id, tissue_name
            logstream.write('%f,%f,%d,%d,%s,%s\n' % (rep_var, err, dataset.tissues[tissue_name].num_patients, 
                                                     dataset.patients[patient_id].num_tissues, patient_id, tissue_name))
            logstream.flush()

        sum_err += err
        sum_err2 += err*err
        print 'error reconstructing %s,%s: %f' % (patient_id, tissue_name, err)
        print 'zero-estimate: %f' % (rep_var)

        weight = rep_var
        val = err/rep_var

        weighted_sum_err += weight * val
        weighted_sum_err2 += weight * val**2
        sum_weight += weight
        max_weight = max(max_weight, weight)

        if sum_weight > max_weight:
            weighted_mean = weighted_sum_err / sum_weight
            weighted_var = (weighted_sum_err2 / sum_weight) - (weighted_sum_err / sum_weight)**2
            unbiased_scale = max_weight / (sum_weight - max_weight)
            estimation_var = weighted_var * unbiased_scale
            estimation_std = estimation_var ** 0.5
            print 'var explained: %f +/- %f%%' % (100. * (1. - weighted_mean), 100.*estimation_std)
        
        if sample_no > 0:
            mean_err = sum_err/(sample_no+1)
            var_err = (sum_err2 - sum_err*sum_err/(sample_no+1))/(sample_no * (sample_no+1))
            print 'total LOOR err = %f +/- %f' % (mean_err, 2*var_err**.5)
            print 'total ZERO err = %f' % (sum_var / (sample_no+1))

        dataset.addValue(patient_id, tissue_name, removed_rep)
        progress.step()

    return sum_err, sum_var


if __name__ == '__main__':
    logging.addNode('loading dataset')
    dataset = getDataset()
    #dataset.addTechnicalsFromFile(args.technical)
    #dataset.regressTechnicals()
    logging.closeNode()
    # TODO specify logstream

    logging.addNode('initilizing logfile')
    logdir = 'results/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = os.path.join(logdir,'i%d-d%d-tin%d-pin%d.csv' % (args.max_iter, args.dimension, int(args.inertia), int(args.inertia)))
    logstream = open(logfile, 'w')
    logstream.write('scale,err,#patients,#tissues,patient_id,tissue_name\n')
    logging.closeNode()
    logging.addNode('running LOOR')
    print "LOOR err: %f\nZERO err: %f" % LeaveOneOutReconstruction(dataset, max_iter=args.max_iter, dimension=args.dimension, 
                                                                   tissue_inertia=args.inertia, patient_inertia=args.inertia,
                                                                   logstream=logstream)
    logging.exit()

    model = PatientModel(max_iter=100)
    model.fit(dataset)

    #tissue_name = dataset.tissues.keys()[0]
    #tissue = dataset.tissues[tissue_name]

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




