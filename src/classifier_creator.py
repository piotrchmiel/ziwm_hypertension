from os import path

from sklearn.tree import DecisionTreeClassifier

from src.factories import LearningSetFactory, MulticlassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, ALGORITHMS, METHODS
from src.utils import save_object


def create_classifiers(method, train_set, train_labels, *args, **kwargs):
    for name, algorithm_class in ALGORITHMS.items():
        if name in METHODS[method]:
            multiclass_classifier = MulticlassClassifierFactory.make_default_classifier(algorithm_class, train_set,
                                                                                        train_labels, *args, **kwargs)
            save_object(path.join(CLASSIFIERS_DIR, "".join(['multiclass_', name, '.pickle'])), multiclass_classifier)

            two_layer_classifier = MulticlassClassifierFactory.make_default_two_layer_classifier(algorithm_class,
                                                                                                 train_set,
                                                                                                 train_labels, *args,
                                                                                                 **kwargs)
            save_object(path.join(CLASSIFIERS_DIR, "".join(['two_layer_', name, '.pickle'])), two_layer_classifier)


def main():
    print("Getting learning sets...")

    train_set, train_labels, test_set, test_labels = LearningSetFactory.get_learning_sets_and_labels(0.8)

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])

    print("Done.")
    print("Creating classifiers...")

    keywords_ensemble = {'n_estimators': 50}
    keywords_multiclass = {'n_jobs': -1}  # -1 means: use all CPUs

    create_classifiers('ensemble', train_set, train_labels, **keywords_ensemble)
    create_classifiers('multiclass', train_set, train_labels, DecisionTreeClassifier(), **keywords_multiclass)

    print("Done.")

if __name__ == '__main__':
    main()
