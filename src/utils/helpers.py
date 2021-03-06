from copy import deepcopy
from cProfile import Profile
from os import path
from pstats import Stats
from time import time
from timeit import timeit

import numpy as np
from joblib import Parallel, delayed
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pyprof2calltree import convert
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import indexable, safe_indexing
from sklearn.utils.validation import _num_samples

from src.classifiers.sklearn_wrapper import SklearnWrapper
from src.factories.classifier_factory import ClassifierFactory
from src.settings import TIME_BENCH_DIR


def cross_val_score(classifiers_to_test, X, y=None, cv=None, factory=ClassifierFactory.make_multiclass_classifier,
                    n_jobs=1):
    X, y = indexable(X, y)

    cv = StratifiedKFold(y, cv, shuffle=True)

    scores = Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(algorithm_info, factory, X, y, train, test)
                                     for algorithm_info in classifiers_to_test.values() for train, test in cv)

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

    train_set = safe_indexing(X, train)
    train_labels = safe_indexing(y, train)

    if 'n_neighbors' in algorithm_info[1]:
        algorithm_info[1]['n_neighbors'] = 3 * len(set(y))

    classifier = factory(algorithm_info[0], train_set, train_labels, **algorithm_info[1])

    test_set = safe_indexing(X, test)
    test_labels = safe_indexing(y, test)
    train_score = classifier.accuracy(test_set, test_labels)
    scoring_time = time() - start_time

    return [train_score, _num_samples(test), scoring_time]


def time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=1, n_jobs=1,
                   profile=False, generate_graph=False):

    cpu = str(n_jobs) + "CPU" if n_jobs > 0 else "all CPUs"
    print(classifier_name, cpu)
    elapsed_timeit = 0
    elapsed_time = 0
    algorithm_info[1]['n_jobs'] = n_jobs

    if 'n_neighbors' in algorithm_info[1]:
        algorithm_info[1]['n_neighbors'] = 3 * len(set(train_labels))

    for _ in range(n_times):

        classifier_timeit = SklearnWrapper(algorithm_info[0](**algorithm_info[1]))
        elapsed_timeit += timeit('classifier.train(X,y)', number=1, globals={'classifier': classifier_timeit,
                                                                             'X': deepcopy(train_set),
                                                                             'y': deepcopy(train_labels)})

        classifier_time = SklearnWrapper(algorithm_info[0](**algorithm_info[1]))
        start = time()
        classifier_time.train(deepcopy(train_set), deepcopy(train_labels))
        elapsed_time += time() - start

    if profile:
        classifier_cprofile = SklearnWrapper(algorithm_info[0](**algorithm_info[1]))
        profiler = Profile()
        profiler.runctx('classifier_cprofile.train(train_set, train_labels)', globals(), locals())
        profiler.dump_stats(path.join(TIME_BENCH_DIR, "".join([classifier_name, '_', str(n_jobs), ".dat"])))
        convert(Stats(profiler), path.join(TIME_BENCH_DIR, "".join([classifier_name, '_', str(n_jobs), ".profile"])))
        print("Dump CProfile data to",TIME_BENCH_DIR, "done.")

    if generate_graph:
        classifier_graph = SklearnWrapper(algorithm_info[0](**algorithm_info[1]))
        with PyCallGraph(output=GraphvizOutput()):
            classifier_graph.train(deepcopy(train_set), deepcopy(train_labels))

    print("Timeit : {0:.6f} s".format(elapsed_timeit/n_times))
    print("Time   : {0:.6f} s".format(elapsed_time/n_times))