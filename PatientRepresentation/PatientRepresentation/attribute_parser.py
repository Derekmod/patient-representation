import os
import attribute_parser as this

_BINARIES = None
_CATEGORICALS = None
_IGNORES = None
_UNINTERESTING = None

def load_list(filename):
    ret = []
    for label in open(filename, 'r'):
        if label:
            ret += [label]
    return ret

def loadBinaries():
    realdir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(realdir, 'binaries.txt')
    this._BINARIES = load_list(filename)
def loadCategoricals():
    realdir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(realdir, 'categoricals.txt')
    this._CATEGORICALS = load_list(filename)
def loadIgnore99():
    realdir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(realdir, 'ignore99s.txt')
    this._IGNORES = load_list(filename)
def loadUninteresting():
    realdir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(realdir, 'uninteresting_attributes.txt')
    this._UNINTERESTING = load_list(filename)
    
def isBinary(label):
    if this._BINARIES is None:
        this.loadBinaries()

    return label in this._BINARIES
def isCategorical(label):
    if this._CATEGORICALS is None:
        this.loadCategoricals()

    return label in this._CATEGORICALS
def isIgnore99(label):
    if this._IGNORES is None:
        this.loadIgnore99()

    return label in this._IGNORES
def isUninteresting(label):
    if this._UNINTERESTING is None:
        this.loadUninteresting()

    return label in this._UNINTERESTING

def regressableSubjectAttributes(filename, sep='\t'):
    stream = open(filename, 'r')
    labels = stream.readline().strip().split(sep)

    categories = dict()  # label --> class --> subjects

    for line in stream:
        items = line.strip().split(sep)
        patient_id = items[0]
        for idx in range(1, len(items)):
            label = labels[idx]
            if not this.isUninteresting(label):
                continue
            if this.isCategorical(label):
                if label not in categories:
                    categories[label] = dict()
                val = items[idx]
                if val not in categories[label]:
                    categories[label][val] = []
                categores[label][val] += [patient_id]

            try:
                val = float(items[idx])
                if this.isIgnore99(label) and val > 90:
                    continue
                yield patient_id, label, val
            except:
                pass

    for label in categories:
        for sublabel in categories[label]:
            for patient_id in categories[label][sublabel]:
                for sublabel2 in categories[label]:
                    val = 0.
                    if sublabel == sublabel2:
                        val = 1.
                    yield patient_id, label + '-' + sublabel2, val

def regressableSampleAttributes(filename, sep='\t'):
    f = open(filename, 'r')
    labels = f.readline().strip().split(sep)

    for line in f:
        items = line.strip().split(sep)

        id = items[0]
        id_components = id.split('-')
        patient_id = '-'.join(id_components[:2])
        tissue_name = items[13].replace(' ', '').replace('-', '_')
        sample_id = patient_id, tissue_name

        for i in range(1, len(items)):
            label = labels[i]
            try:
                val = float(items[i])
                if this.isIgnore99(label) and val > 90:
                    continue
                yield sample_id, label, val
            except:
                pass

def learnableSubjectAttributes(filename, sep='\t'):
    f = open(filename, 'r')
    label = f.readline().strip().split(sep)

    for line in f:
        items = line.strip().split(sep)
        patient_id = items[0]

        for idx in range(1, len(items)):
            label = labels[idx]
            if this.isUninteresting(label):
                continue
            if this.isCategorical(label):
                if label not in categories:
                    categories[label] = dict()
                val = items[idx]
                if val not in categories[label]:
                    categories[label][val] = []
                categores[label][val] += [patient_id]

            try:
                val = float(items[idx])
                if this.isIgnore99(label) and val > 90:
                    continue
                yield patient_id, label, val
            except:
                pass

    for label in categories:
        for val, sublabel in enumerate(categories[label]):
            for patient_id in categories[label][sublabel]:
                yield patient_id, label, val