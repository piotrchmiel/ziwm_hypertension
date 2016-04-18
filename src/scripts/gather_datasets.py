from glob import glob
from os import sep, remove
from os.path import dirname, exists, join as pjoin, split, splitext, realpath
from shutil import move
from urllib.request import urlretrieve

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
        'files': ['tctodd' + str(i) for i in range(1, 10)],
        'operation': 'auslan_concat',
        'out': 'auslan.csv'
    },
    'isolet': {
        'download': [
            'http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z',
            'http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z'
        ],
        'files': [
            'isolet1+2+3+4.data',
            'isolet5.data'
        ],
        'operation': 'concat',
        'out': 'isolet.csv'
    },
    'abalone': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/abalone.zip'
        ],
        'files': ['abalone.dat'],
        'operation': 'dat2csv',
        'out': 'abalone.csv'
    },
    'letter': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/letter.zip'
        ],
        'files': ['letter.dat'],
        'operation': 'dat2csv',
        'out': 'letter.csv'
    },
    'kddcup': {
        'download': [
            'http://sci2s.ugr.es/keel/dataset/data/classification/kddcup.zip'
        ],
        'files': ['kddcup.dat'],
        'operation': 'dat2csv',
        'out': 'kddcup.csv'
    },
}


def main():
    training_root = pjoin(split(dirname(dirname(realpath(__file__))))[0], 'Training set')

    for dataset, items in datasets.items():
        print('Getting dataset', dataset.upper(), 'if needed...')
        # download datasets
        for link in items['download']:
            filename = link.split('/')[-1]
            archive = pjoin(training_root, filename)
            if not exists(archive):
                print('Downloading', filename)
                urlretrieve(link, archive)

        # extract files
        for out_file in items['files']:
            if not exists(pjoin(training_root, out_file)):
                links = [d for d in items['download'] if splitext(out_file)[0] in d]
                if len(links) > 0:
                    filename = links[0].split('/')[-1]
                    archive = pjoin(training_root, filename)
                else:
                    archive = pjoin(training_root, items['download'][0].split('/')[-1])
                print('Extracting', archive)
                if archive.endswith('.tar.gz'):
                    extract_archive(archive, outdir=dirname(archive), program='py_tarfile',
                                    verbosity=-1, interactive=False)
                else:
                    extract_archive(archive, outdir=dirname(archive), verbosity=-1, interactive=False)

        # generate output files
        if type(items['files']) == dict:
            print('Renaming files')
            for source, destination in items['files'].items():
                if not exists(pjoin(training_root, destination)):
                    move(pjoin(training_root, source), pjoin(training_root, destination))
        out_filename = pjoin(training_root, items['out'])
        print('Generating output files')
        if items['operation'] == 'auslan_concat':
            first_record = True
            with open(out_filename, 'w') as out_file:
                for filename in glob(pjoin(training_root, 'tctodd*', '*')):
                    class_ = filename.split(sep)[-1].split('-')[0]
                    with open(pjoin(training_root, filename)) as infile:
                        for line in infile:
                            if first_record:
                                header = ';'.join(
                                    ['Atr-' + str(i) for i in range(1, len(line.split('\t')) + 1)]
                                ) + ';Class'
                                out_file.write(header + '\n')
                                first_record = False
                            out_file.write(line.replace('\t', ';').rstrip() + ';' + class_ + '\n')
        elif not exists(out_filename):
            if items['operation'] == 'concat' or items['operation'] == 'dat2csv':
                first_record = True
                with open(out_filename, 'w') as out_file:
                    for fname in items['files']:
                        with open(pjoin(training_root, fname)) as infile:
                            for line in infile:
                                if first_record and items['operation'] == 'concat':
                                    first_record = False
                                    header = ';'.join(
                                        ['Atr-' + str(i) for i in range(1, len(line.split(', ')))]
                                    ) + ';Class'
                                    out_file.write(header + '\n')
                                out_file.write(line.replace(', ', ';').replace(',', ';'))
            if items['operation'] == 'dat2csv':
                with open(out_filename, 'r+') as out_file:
                    with open(out_filename + '.tmp', 'w') as new_file:
                        skip_contents = True
                        header = ''
                        for line in out_file:
                            if not '@data' in line:
                                if not skip_contents:
                                    new_file.write(line)
                                else:
                                    if '@inputs' in line or '@outputs' in line:
                                        header += line.split(' ')[1].rstrip() + ';'
                            else:
                                skip_contents = False
                                new_file.write(header[:-1] + '\n')
                remove(out_filename)
                move(out_filename + '.tmp', out_filename)
        print('Done.')

if __name__ == '__main__':
    main()
