from os.path import dirname, isfile, join as pjoin, split, realpath
from urllib.request import urlretrieve
from patoolib import extract_archive


def main():
    datasets = {
        'mnist': [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ],
        'isolet': ['https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz'],
        'auslan': [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z'
        ],
        'abalone': ['http://sci2s.ugr.es/keel/dataset/data/classification/abalone.zip'],
        'letter': ['http://sci2s.ugr.es/keel/dataset/data/classification/letter.zip'],
        'kddcup': ['http://sci2s.ugr.es/keel/dataset/data/classification/kddcup.zip'],
    }
    for dataset, links in datasets.items():
        print('Getting dataset', dataset, 'if needed...')
        for link in links:
            filename = link.split('/')[-1]
            file = pjoin(split(dirname(realpath(__file__)))[0], 'Training set', filename)
            if not isfile(file):
                print('Downloading', filename)
                urlretrieve(link, file)
            if not file.endswith('.tar.gz'):
                extract_archive(file, outdir=dirname(file), verbosity=-1, interactive=False)
            else:
                extract_archive(file, outdir=dirname(file), program='py_tarfile', verbosity=-1, interactive=False)
        print('Done.')

if __name__ == '__main__':
    main()
