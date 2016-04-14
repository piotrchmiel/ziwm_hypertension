from itertools import chain
from os import path

from joblib import Parallel, delayed

from src.factories.classifier_factory import ClassifierFactory
from src.factories.learning_factory import LearningSetFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, METHODS, MULTICLASS, ENSEMBLE
from src.utils.tools import save_object, get_arguments


def get_creator(method, train_set, train_labels):
    return (delayed(create_classifiers)(method, classifier_name, algorithm_info, train_set, train_labels)
            for classifier_name, algorithm_info in METHODS[method].items())


def create_classifiers(method, classifier_name, algorithm_info, train_set, train_labels):
    multiclass_classifier = ClassifierFactory.make_multiclass_classifier(algorithm_info[0], train_set, train_labels,
                                                                         **algorithm_info[1])

    save_object(path.join(CLASSIFIERS_DIR, method, "".join(['multiclass_', classifier_name, '.pickle'])),
                multiclass_classifier)

    if method == ENSEMBLE:
        two_layer_classifier = ClassifierFactory.make_two_layer_classifier(algorithm_info[0], train_set, train_labels,
                                                                           **algorithm_info[1])

        save_object(path.join(CLASSIFIERS_DIR, ENSEMBLE,  "".join(['two_layer_', classifier_name, '.pickle'])),
                    two_layer_classifier)


def main():
    args = get_arguments("Classifier Creator")

    print("Getting learning sets...")

    train_set, train_labels, test_set, test_labels = \
        LearningSetFactory.get_learning_sets_and_labels(0.8)

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])

    print("Done.")
    print("Creating classifiers...")

    if args.method == MULTICLASS:
        Parallel(n_jobs=-args.n_jobs)(get_creator(MULTICLASS, train_set, train_labels))
    elif args.method == ENSEMBLE:
        Parallel(n_jobs=-args.n_jobs)(get_creator(ENSEMBLE, train_set, train_labels))
    elif args.method == 'all':
        Parallel(n_jobs=-args.n_jobs)(chain(get_creator(MULTICLASS, train_set, train_labels),
                                            get_creator(ENSEMBLE, train_set, train_labels)))

    print("Done.")

if __name__ == '__main__':
    main()
