import os
import sys
import collapse_logging as logging

import argparse

import dataset as dataset_m

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

def main():
    pickle_filename = os.path.join(args.pickle_dir, 'dataset.pickle')
    if os.path.exists(pickle_filename):
        return

    dataset = getDataset()
    dataset.pickle(pickle_filename)




if __name__ == '__main__':
    main()