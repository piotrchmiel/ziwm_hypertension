from sklearn.tree import DecisionTreeClassifier

from src.factories import LearningSetFactory, MulticlassClassifierFactory
from src.settings import ALGORITHMS, METHODS


def main():
    iterations = 2
    keywords_ensemble = {}
    keywords_multiclass = {'n_jobs': -1}  # -1 means: use all CPUs

    print("Starting Test..")

    results = {}
    for name, _ in ALGORITHMS.items():
        results.update({'multiclass_%s' % name: 0})
        results.update({'two_layer_%s' % name: 0})

    for i in range(iterations):
        print(i)
        train_set, train_labels, test_set, test_labels = LearningSetFactory.get_learning_sets_and_labels(0.8)
        for name, algorithm in ALGORITHMS.items():
            if name in METHODS['ensemble']:
                multiclass_classifier = \
                    MulticlassClassifierFactory.make_default_classifier(algorithm, train_set, train_labels,
                                                                        **keywords_ensemble)
                two_layer_classifier = \
                    MulticlassClassifierFactory.make_default_two_layer_classifier(algorithm, train_set, train_labels,
                                                                                  **keywords_ensemble)
            else:
                multiclass_classifier = \
                    MulticlassClassifierFactory.make_default_classifier(algorithm, train_set, train_labels,
                                                                        DecisionTreeClassifier(),
                                                                        **keywords_multiclass)
                two_layer_classifier = \
                    MulticlassClassifierFactory.make_default_two_layer_classifier(algorithm, train_set, train_labels,
                                                                                  DecisionTreeClassifier(),
                                                                                  **keywords_multiclass)

            results['multiclass_%s' % name] += multiclass_classifier.accuracy(test_set, test_labels)
            results['two_layer_%s' % name] += two_layer_classifier.accuracy(test_set, test_labels)

    for name, result in results.items():
        print(name.ljust(32), ":", round(result/iterations, 2))

    print("Done.")

if __name__ == '__main__':
    main()
