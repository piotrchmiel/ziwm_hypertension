from argparse import ArgumentParser

from joblib import dump, load


def save_object(file_location, class_object):
    dump(class_object, file_location, 3)


def load_object(file_location):
    return load(file_location)


def get_neighbors_above_threshold(y, neighbors, threshold):
    neighbors_list = []

    for neighbor in neighbors[0]:
        neighbors_list.append(y[neighbor])

    neighbors_count = len(neighbors_list)
    neighbors_set = set(neighbors_list)
    neighbors_set_tmp = neighbors_set.copy()
    for neighbor in neighbors_set_tmp:
        if not neighbors_list.count(neighbor) / neighbors_count > threshold:
            neighbors_set.remove(neighbor)

    return neighbors_set


def get_arguments(script_name):
    parser = ArgumentParser(script_name)
    parser.add_argument('--n-jobs', '-n', default=-1, type=int, help="Number of used CPU cores. Default: all cores")
    parser.add_argument('--method', '-m', default='multiclass',
                        choices=['multiclass', 'ensemble', 'all'],
                        type=str, help="This parameter determines which classifiers will be created.")

    return parser.parse_args()


