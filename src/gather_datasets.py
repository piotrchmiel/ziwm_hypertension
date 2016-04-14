from os import remove
from os.path import dirname, isfile, join as pjoin, split, splitext, realpath
from urllib.request import urlretrieve
from shutil import move
from patoolib import extract_archive

datasets = {
    'mnist': {
        'download': [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ],
        'files': {
            'train-images.idx3-ubyte': 'train-images-idx3-ubyte',
            'train-labels.idx1-ubyte': 'train-labels-idx1-ubyte',
            't10k-images.idx3-ubyte': 't10k-images-idx3-ubyte',
            't10k-labels.idx1-ubyte': 't10k-labels-idx1-ubyte'
        },
        'operation': 'nop',
        'out': ''
    },
    'auslan': {
        'download': [
            'http://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz'
        ],
        'files': {},
        'operation': 'nop',
        'out': ''
    },
    'isolet': {
        'download': [
            'http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z',
            'http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z'
        ],
        'files': {
            'isolet1+2+3+4.data',
            'isolet5.data'
        },
        'operation': 'concat',
        'out': 'isolet.csv'
    },
    'abalone': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/abalone.zip'
        ],
        'files': {'abalone.dat'},
        'operation': 'dat2csv',
        'out': 'abalone.csv'
    },
    'letter': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/letter.zip'
        ],
        'files': {'letter.dat'},
        'operation': 'dat2csv',
        'out': 'letter.csv'
    },
    'kddcup': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/kddcup.zip'
        ],
        'files': {'kddcup.dat'},
        'operation': 'dat2csv',
        'out': 'kddcup.csv'
    },
}


def main():
    training_root = pjoin(split(dirname(realpath(__file__)))[0], 'Training set')
    for dataset, items in datasets.items():
        print('Getting dataset', dataset.upper(), 'if needed...')
        # download datasets
        for link in items['download']:
            filename = link.split('/')[-1]
            archive = pjoin(training_root, filename)
            if not isfile(archive):
                print('Downloading', filename)
                urlretrieve(link, archive)

        # extract files
        for file in items['files']:
            if not isfile(pjoin(training_root, file)):
                links = [d for d in items['download'] if splitext(file)[0] in d]
                if len(links) > 0:
                    filename = links[0].split('/')[-1]
                    archive = pjoin(training_root, filename)
                    if not file.endswith('.tar.gz'):
                        extract_archive(archive, outdir=dirname(archive), verbosity=-1, interactive=False)
                    else:
                        extract_archive(archive, outdir=dirname(archive), program='py_tarfile',
                                        verbosity=-1, interactive=False)

        # generate output files
        if type(items['files']) == dict:
            for source, destination in items['files'].items():
                move(pjoin(training_root, source), pjoin(training_root, destination))
        out_filename = pjoin(training_root, items['out'])
        if items['operation'] == 'nop':
            pass
        elif isfile(out_filename):
            if items['operation'] == 'concat' or items['operation'] == 'dat2csv':
                with open(out_filename, 'w') as file:
                    for fname in items['files']:
                        with open(pjoin(training_root, fname)) as infile:
                            for line in infile:
                                file.write(line.replace(', ', ';').replace(',', ';'))
            if items['operation'] == 'dat2csv':
                with open(out_filename, 'r+') as file:
                    with open(out_filename + '.tmp', 'w') as new_file:
                        skip_contents = True
                        header = ''
                        for line in file:
                            if not '@data' in line:
                                if not skip_contents:
                                    new_file.write(line)
                                else:
                                    if '@inputs' in line or '@outputs' in line:
                                        header += line.split(' ')[1].rstrip() + ";"
                            else:
                                skip_contents = False
                                new_file.write(header[:-1] + '\n')
                remove(out_filename)
                move(out_filename + '.tmp', out_filename)
        print('Done.')

if __name__ == '__main__':
    main()
