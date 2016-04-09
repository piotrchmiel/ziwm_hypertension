from os import path

from src.factories import LearningSetFactory, MultiClassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, ALGORITHMS
from src.utils import save_object


def main():
    print("Getting learning sets...")

    train_set, train_labels, test_set, test_labels = LearningSetFactory.get_learning_sets_and_labels(0.8)

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])

    print("Done.")
    print("Creating classifiers...")

    keywords_ensemble = {'n_estimators': 50}

    for name, algorithm_class in ALGORITHMS.items():
        multiclass_classifier = MultiClassClassifierFactory.make_default_classifier(algorithm_class, train_set,
                                                                                    train_labels, **keywords_ensemble)
        save_object(path.join(CLASSIFIERS_DIR, "".join(['multiclass_', name, '.pickle'])), multiclass_classifier)

        two_layer_classifier = MultiClassClassifierFactory.make_default_two_layer_classifier(algorithm_class, train_set,
                                                                                             train_labels,
                                                                                             **keywords_ensemble)
        save_object(path.join(CLASSIFIERS_DIR, "".join(['two_layer_', name, '.pickle'])), two_layer_classifier)

    print("Done.")

if __name__ == '__main__':
    main()
