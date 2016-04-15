from time import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

from src.factories.classifier_factory import ClassifierFactory


def cross_val_score(classifiers_to_test, X, y=None, cv=None, factory=ClassifierFactory.make_multiclass_classifier,
                    n_jobs=1):
    X, y = indexable(X, y)

    cv = StratifiedKFold(y, cv, shuffle=True)

    scores = Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(algorithm_info, factory, X, y, train, test)
                                     for algorithm_info in classifiers_to_test.values() for train, test in cv )

    final_results = []

    for number, classifier_name in enumerate(classifiers_to_test.keys()):
        classifier_score = np.array(scores[number*len(cv):(number+1)*len(cv)])
        final_results.append("Accuracy {0}: {1:.3f} (+/- {2:.3f}), Samples {3:.1f} "
                             "(+/- {4:.3f}), Time {5:.3f} (+/- {6:.3f})".format(
            classifier_name, classifier_score[:, 0].mean(), classifier_score[:, 0].std() * 2,
            classifier_score[:, 1].mean(), classifier_score[:, 1].std() * 2,
            classifier_score[:, 2].mean(), classifier_score[:, 2].std() * 2))

    return '\n'.join(sorted(final_results))


def fit_and_score(algorithm_info, factory, X, y, train, test):
    start_time = time()
    train_set = (X[number] for number in train)
    train_labels = [y[number] for number in train]
    classifier = factory(algorithm_info[0], train_set, train_labels,
                                                              **algorithm_info[1])
    test_set = (X[number] for number in test)
    test_labels = [y[number] for number in test]
    train_score = classifier.accuracy(test_set, test_labels)
    scoring_time = time() - start_time

    return [train_score, _num_samples(test), scoring_time]
