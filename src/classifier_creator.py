from os import path

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from src.factories import LearningSetFactory, MultiClassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR
from src.utils import save_object


def main():
    print("Getting learning sets...")

    train_set, train_labels, test_set, test_labels = LearningSetFactory.get_learning_sets_and_labels(0.8)

    print("Done.")
    print("Creating classifiers...")

    multiclass_random_forest = MultiClassClassifierFactory.make_default_classifier(RandomForestClassifier, train_set,
                                                                                   train_labels)
    multiclass_ada_boost_classifier = MultiClassClassifierFactory.make_ada_boost_classifier(train_set, train_labels)
    multiclass_bagging_classifier = MultiClassClassifierFactory.make_default_classifier(BaggingClassifier,
                                                                                                train_set, train_labels)

    two_layer_random_forest = MultiClassClassifierFactory.make_default_two_layer_classifier(RandomForestClassifier,
                                                                                              train_set, train_labels)
    two_layer_ada_boost_classifier = MultiClassClassifierFactory.make_ada_boost_two_layer_classifier(train_set,
                                                                                                     train_labels)
    two_layer_bagging_classifier = MultiClassClassifierFactory.make_default_two_layer_classifier(BaggingClassifier,
                                                                                                train_set, train_labels)
    print("Done.")
    print("Saving classifiers...")

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])
    save_object(path.join(CLASSIFIERS_DIR, "multiclass_random_forest.pickle"), multiclass_random_forest)
    save_object(path.join(CLASSIFIERS_DIR, "multiclass_ada_boost_classifier.pickle"), multiclass_ada_boost_classifier)
    save_object(path.join(CLASSIFIERS_DIR, "multiclass_bagging_classifier.pickle"), multiclass_bagging_classifier)
    save_object(path.join(CLASSIFIERS_DIR, "two_layer_random_forest.pickle"), two_layer_random_forest)
    save_object(path.join(CLASSIFIERS_DIR, "two_layer_ada_boost_classifier.pickle"), two_layer_ada_boost_classifier)
    save_object(path.join(CLASSIFIERS_DIR, "two_layer_bagging_classifier.pickle"), two_layer_bagging_classifier)

    print("Done.")

if __name__ == '__main__':
    main()
