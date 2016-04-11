from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from src.factories import LearningSetFactory, ClassifierFactory
from src.settings import ALGORITHMS, METHODS


def benchmark_result(algorithm_name, algorithm, sets):
    keywords = {'n_estimators': 50} if algorithm_name in METHODS['ensemble'] else {'n_jobs': -1,
                                                                                   'estimator':
                                                                                       DecisionTreeClassifier()}

    multiclass_classifier_accuracy = ClassifierFactory.make_multiclass_classifier(algorithm, sets[0], sets[1],
                                                                                  **keywords).accuracy(sets[2], sets[3])
    two_layer_classifier_accuracy = ClassifierFactory.make_two_layer_classifier(algorithm, sets[0], sets[1],
                                                                                **keywords).accuracy(sets[2], sets[3])

    return algorithm_name, multiclass_classifier_accuracy, two_layer_classifier_accuracy


def main():
    iterations = 1
    accuracy_results = {}
    print("Starting Test..")

    for i in range(iterations):
        train_and_test_sets = LearningSetFactory.get_learning_sets_and_labels(0.8)

        result = Parallel(n_jobs=-1)(delayed(benchmark_result)(name, algorithm, train_and_test_sets)
                            for name, algorithm in ALGORITHMS.items())

        for name, multi_accurancy, twol_accuracy in result:
            accuracy_results['multiclass_%s' % name] = accuracy_results.get('multiclass_%s' % name, 0) + multi_accurancy
            accuracy_results['two_layer_%s' % name] = accuracy_results.get('two_layer_%s' % name, 0) + twol_accuracy

    for name, result in accuracy_results.items():
        print(name.ljust(32), ":", round(result/iterations, 2))

    print("Done.")

if __name__ == '__main__':
    main()
