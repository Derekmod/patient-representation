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
parser.add_argument("--logfile", type=str)
parser.add_argument("--split-seed", type=int)
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
if args.split_seed is None:
    args.split_seed = 5318008


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
        logging.log('after data: ' + dataset.summary)

        dataset.normalize(verbose=True)
        logging.log('after normalize: ' + dataset.summary)

        technical_labels = dataset.addTechnicalsFromFile(args.technicals, regress=False, verbose=True)
        covariate_labels = dataset.addCovariatesFromFile(args.covariates, verbose=True)
        dataset.regressCovariates(cov_names=technical_labels, verbose=True)
        logging.log('after regress: ' + dataset.summary)


        dataset.runPCA(var_per_dim=10, verbose=True)
        logging.log('after PCA: ' + dataset.summary)

        logging.log('pickling')
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

def EvaluateCovariates(dataset, train_ids, test_ids, max_iter=1000, dimension=6, tissue_inertia=5., patient_inertia=5., lam=5.):
    logging.log('training model')
    model = PatientModel(max_iter=max_iter, weight_inertia=tissue_inertia)
    model.fit(dataset)

    logging.addNode('loading covariate auxiliary data')
    realdir = os.path.dirname(os.path.realpath(__file__))
    logging.log('loading ignore 99')
    ignore_fn = os.path.join(realdir, 'ignore99s.txt')
    ignore_labels = []
    for line in open(ignore_fn, 'r'):
        if not line:
            break
        ignore_labels += [line]
    logging.log('loading categoricals')
    categoricals_fn = os.path.join(realdir, 'categoricals.txt')
    categories = dict()
    for line in open(categoricals_fn, 'r'):
        if not line:
            break
        categories[line] = dict()
    logging.closeNode()

    logging.addNode('loading covariates')
    stream = open(args.covariates, 'r')
    labels = stream.readline().strip().split('\t')
    covariates = dict()
    for line in stream:
        items = line.strip().split('\t')
        patient_id = items[0]
        for idx in range(1, len(items)):
            val = items[idx]
            label = labels[idx]
            if label in categories:
                if val not in categories[label]:
                    categories[label][val] = []
                categories[label][val] += [patient_id]
                continue
            try:
                val = float(items[idx])
                if label in ignore_labels and val > 90:
                    continue
                if label not in covariates:
                    covariates[label] = dict()
                covariates[label][patient_id] = val
            except:
                continue
    logging.closeNode()


    logging.addNode('learning covariates')
    results = []
    for label in covariates:
        trainX_raw = []
        trainY_raw = []
        testX_raw = []
        testY_raw = []

        for patient_id in covariates[label]:
            if patient_id in train_ids:
                trainX_raw += [model.patient_reps[patient_id]]
                trainY_raw += [covariates[label][patient_id]]
            elif patient_id in test_ids:
                testX_raw += [model.patient_reps[patient_id]]
                testY_raw += [covariates[label][patient_id]]
            else:
                #logging.log('id %s not found' % patient_id)
                pass

        if not trainX_raw or not testX_raw:
            logging.log('label "%s" had insufficient samples' % label)
            continue
                
        trainX = np.concatenate(trainX_raw, axis=0)
        trainY = np.array([trainY_raw]).T
        testX = np.concatenate(testX_raw, axis=0)
        testY = np.array([testY_raw]).T

        ntrain, dim = trainX.shape

        meanX, meanY = trainX.mean(), trainY.mean()
        residX = trainX - meanX
        residY = trainY - meanY

        weights = np.linalg.inv(residX.T.dot(residX) + dim * lam * np.eye(dim)).dot(residX.T).dot(residY)

        pred = meanY + (testX - meanX).dot(weights)

        naive_err = (testY - trainY.mean()).T.dot(testY - trainY.mean())
        trained_err = (testY - pred).T.dot(testY - pred)
        var_exp = (naive_err - trained_err)/naive_err
        logging.log('on "%s": naive=%f, trained=%f, var_exp=%f, #samples=%f,%f' % (label, naive_err, trained_err, var_exp, len(trainX_raw), len(testX_raw)))
        results += [(var_exp, label)]
    logging.closeNode()

    results.sort()
    logging.addNode('results')
    for var_exp, label in results:
        logging.log('%s\t: %f' % (label, var_exp))
    logging.closeNode()


def trainHyperParams(dataset, train_samples=None, lr=0.2, asym_epoch=1000, tissue_inertia=2., patient_inertia=2., lam=5., dim=6.):
    logging.addNode('training hypers')

    

    logging.closeNode()




if __name__ == '__main__':
    if args.logfile:
        logging.set_logfile(args.logfile)
    else:
        logging.set_logfile('log.out')

    logging.addNode('loading dataset')
    dataset = getDataset()
    logging.closeNode()


    #train_samples, val_samples, test_samples = dataset.split()
    patient_ids = dataset.patients.keys()
    np.random.seed(args.split_seed)
    patient_ids = np.random.permutation(patient_ids)
    npatients = len(patient_ids)
    ntrain = int(npatients * 0.6)
    nvalid = int(npatients * (0.6 + 0.2))
    train_ids, val_ids = patient_ids[:ntrain], patient_ids[ntrain:nvalid]



    #logging.addNode('initilizing logfile')
    #logdir = 'results/'
    #if not os.path.exists(logdir):
    #    os.makedirs(logdir)
    #logfile = os.path.join(logdir,'i%d-d%d-tin%d-pin%d.csv' % (args.max_iter, args.dimension, int(args.inertia), int(args.inertia)))
    #logstream = open(logfile, 'w')
    #logstream.write('scale,err,#patients,#tissues,patient_id,tissue_name\n')
    #logging.closeNode()
    #logging.addNode('running LOOR')
    #print "LOOR err: %f\nZERO err: %f" % LeaveOneOutReconstruction(dataset, max_iter=args.max_iter, dimension=args.dimension, 
    #                                                               tissue_inertia=args.inertia, patient_inertia=args.inertia,
    #                                                               logstream=logstream)
    #logging.exit()

    EvaluateCovariates(dataset, train_ids, val_ids, max_iter=100)
    logging.exit()
    model = PatientModel(max_iter=1000)
    model.fit(dataset)
    

    #tissue_name = dataset.tissues.keys()[0]
    #tissue = dataset.tissues[tissue_name]

    # print tissue.value[tissue.rows[patient_id],:]
    # print model.predict(patient_id, tissue_name)

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




