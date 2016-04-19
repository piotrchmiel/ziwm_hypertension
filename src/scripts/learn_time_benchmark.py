from itertools import chain
from warnings import filterwarnings

from src.factories.learning_factory import LearningSetFactory
from src.settings import MULTICLASS, ENSEMBLE, METHODS
from src.utils.helpers import time_benchmark
from src.utils.tools import get_arguments


def main():
    args = get_arguments("Time Benchmark")
    filterwarnings("ignore")

    print("Getting learning sets, using:", args.dataset.upper())

    train_set, train_labels = LearningSetFactory.get_full_learning_set_with_labels(
        getattr(LearningSetFactory.DataSource, args.dataset))

    if args.method == MULTICLASS:
        for classifier_name, algorithm_info in METHODS[MULTICLASS].items():
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=-1)
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=1)
    elif args.method == ENSEMBLE:
        for classifier_name, algorithm_info in METHODS[ENSEMBLE].items():
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=-1)
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=1)
    elif args.method == 'all':
        for classifier_name, algorithm_info in chain(METHODS[MULTICLASS].items(), METHODS[ENSEMBLE].items()):
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=-1)
            time_benchmark(classifier_name, algorithm_info, train_set, train_labels, n_times=10, n_jobs=1)

if __name__ == '__main__':
    main()
