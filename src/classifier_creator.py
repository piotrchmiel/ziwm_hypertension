from argparse import ArgumentParser
from itertools import chain
from os import path

from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from src.factories import LearningSetFactory, ClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, ALGORITHMS, METHODS
from src.utils import save_object


def get_arguments():
    parser = ArgumentParser("Classifier")
    parser.add_argument('--n-jobs', '-n', default=-1, type=int, help="Number of used CPU cores. Default: all cores")
    parser.add_argument('--method', '-m', default='dynamic_multiclass',
                        choices=['dynamic_multiclass', 'ensemble', 'all'],
                        type=str, help="This parameter determines which classifiers will be created.")

    return parser.parse_args()


def create_classifiers(name, algorithm_class, train_set, train_labels, *args, **kwargs):
    multiclass_classifier = ClassifierFactory.make_multiclass_classifier(algorithm_class,
                                                                         train_set,
                                                                         train_labels,
                                                                         *args, **kwargs)

    save_object(path.join(CLASSIFIERS_DIR, "".join(['multiclass_', name, '.pickle'])),
                multiclass_classifier)

    two_layer_classifier = ClassifierFactory.make_two_layer_classifier(
        algorithm_class, train_set, train_labels, *args, **kwargs)

    save_object(path.join(CLASSIFIERS_DIR, "".join(['two_layer_', name, '.pickle'])),
                two_layer_classifier)


def main():
    args = get_arguments()

    print("Getting learning sets...")

    train_set, train_labels, test_set, test_labels = \
        LearningSetFactory.get_learning_sets_and_labels(0.8)

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])

    print("Done.")
    print("Creating classifiers...")

    keywords_ensemble = {'n_estimators': 50}
    keywords_multiclass = {'n_jobs': -1}  # -1 means: use all CPUs

    multiclass_creator = (delayed(create_classifiers)
                            (name, algorithm_class, train_set, train_labels, DecisionTreeClassifier(),
                              **keywords_multiclass)
                              for name, algorithm_class in ALGORITHMS.items()
                              if name in METHODS['multiclass'])

    ensemble_creator = (delayed(create_classifiers)
                        (name, algorithm_class, train_set, train_labels, **keywords_ensemble)
                          for name, algorithm_class in ALGORITHMS.items()
                          if name in METHODS['ensemble'])

    if args.method == 'dynamic_multiclass':
        Parallel(n_jobs=-args.n_jobs)(multiclass_creator)

    elif args.method == 'ensemble':
        Parallel(n_jobs=-args.n_jobs)(ensemble_creator)
    elif args.method == 'all':
        Parallel(n_jobs=-args.n_jobs)(chain(ensemble_creator, multiclass_creator))

    print("Done.")

if __name__ == '__main__':
    main()
