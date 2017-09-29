import os

import dataset as dataset_m


if __name__ == '__main__':
    basedir = os.path.dirname(os.getcwd())
    basedir = os.path.dirname(basedir)

    data_dir = os.path.join(base_dir, 'data')
    data_dir = os.path.join(base_dir, 'V7 Data')

    dataset = dataset_m.loadFromDir(data_dir, verbose=True)
    print dataset
