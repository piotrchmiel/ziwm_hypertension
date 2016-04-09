from os import path

from src.factories import MultiClassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, ALGORITHMS
from src.utils import load_object


def main():
    print("Loading test set...")

    test_set, test_labels = load_object(path.join(TRAINING_SET_DIR, 'test_set.pickle'))

    print("Benchmark")

    for name, algorithm in ALGORITHMS.items():
        multiclass_classifier = MultiClassClassifierFactory.make_classifier_from_file(path.join(
                CLASSIFIERS_DIR, "".join(['multiclass_', name, '.pickle'])))
        print("{0:25} : {1:.3f}".format("".join(['multiclass_', name]), multiclass_classifier.accuracy(test_set,
                                                                                                       test_labels)))

        two_layer_classifier =  MultiClassClassifierFactory.make_classifier_from_file(path.join(
                CLASSIFIERS_DIR, "".join(['two_layer_', name, '.pickle'])))
        print("{0:25} : {1:.3f}".format("".join(['two_layer_', name]), two_layer_classifier.accuracy(test_set,
                                                                                                     test_labels)))

    print("Done.")


if __name__ == '__main__':
    main()
