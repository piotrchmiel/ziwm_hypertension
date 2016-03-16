from os import path

from src.factories import MultiClassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR
from src.utils import load_object


def main():
    print("Loading test set...")

    test_set, test_labels = load_object(path.join(TRAINING_SET_DIR, 'test_set.pickle'))

    print("Done.\nLoading classifiers...")

    multiclass_random_forest = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                                   "multiclass_random_forest.pickle"))
    multiclass_ada_boost_classifier = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                            "multiclass_ada_boost_classifier.pickle"))
    multiclass_bagging_classifier = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                                "multiclass_bagging_classifier.pickle"))
    two_layer_random_forest = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                                   "two_layer_random_forest.pickle"))
    two_layer_ada_boost_classifier = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                            "two_layer_ada_boost_classifier.pickle"))
    two_layer_bagging_classifier = MultiClassClassifierFactory.make_classifier_from_file(path.join(CLASSIFIERS_DIR,
                                                                                "two_layer_bagging_classifier.pickle"))
    print("Done.\nBenchmark:")

    print("Multiclass Random Forest Classifier :", multiclass_random_forest.accuracy(test_set, test_labels))
    print("Multiclass Ada Boost Classifier     :", multiclass_ada_boost_classifier.accuracy(test_set, test_labels))
    print("Multiclass Bagging Classifier       :", multiclass_bagging_classifier.accuracy(test_set, test_labels))
    print("Two Layer Random Forest Classifier  :", two_layer_random_forest.accuracy(test_set, test_labels))
    print("Two Layer Ada Boost Classifier      :", two_layer_ada_boost_classifier.accuracy(test_set, test_labels))
    print("Two Layer Bagging Classifier        :", two_layer_bagging_classifier.accuracy(test_set, test_labels))

if __name__ == '__main__':
    main()
