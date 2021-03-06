from argparse import ArgumentParser

from joblib import dump, load


def save_object(file_location, class_object):
    dump(class_object, file_location, 3)


def load_object(file_location):
    return load(file_location)


def get_neighbors_above_threshold(labels, neighbors_all, threshold):
    all_neighbors_classes = [labels[neighbor] for neighbor in neighbors_all]

    neighbors_count = len(all_neighbors_classes)
    unique_neighbors_classes = set(all_neighbors_classes)

    return {neighbor for neighbor in unique_neighbors_classes
            if all_neighbors_classes.count(neighbor) / neighbors_count > threshold}


def get_arguments(script_name):
    parser = ArgumentParser(script_name)
    parser.add_argument('--n-jobs', '-n', default=-1, type=int, help="Number of used CPU cores. Default: all cores")
    parser.add_argument('--method', '-m', default='multiclass',
                        choices=['multiclass', 'ensemble', 'all'],
                        type=str, help="This parameter determines which classifiers will be created.")
    parser.add_argument('--dataset', '-d', default='hypertension',
                        choices=['abalone', 'auslan', 'adult', 'ecoli', 'flare', 'hypertension', 'isolet',
                                 'kddcup', 'letter', 'mnist', 'satimage', 'segment', 'shuttle',
                                 'student_alcohol_consumption', 'vowel', 'wine_quality', 'yeast'],
                        type=str, help="This parameter determines which dataset will be used for learning.")

    return parser.parse_args()
