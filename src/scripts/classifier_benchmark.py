from itertools import chain
from os import path, listdir
from warnings import filterwarnings

from joblib import Parallel, delayed

from src.factories.classifier_factory import ClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR, MULTICLASS, ENSEMBLE
from src.utils.tools import load_object, get_arguments


def benchmark_result(classifier_path, test_set, test_labels):
    classifier = ClassifierFactory.make_classifier_from_file(classifier_path)
    return "{0:25} : {1:.3f}".format(path.splitext(path.basename(classifier_path))[0],
                                     classifier.accuracy(test_set, test_labels))


def get_benchmark(method, test_set, test_labels):
    return (delayed(benchmark_result)(path.join(CLASSIFIERS_DIR, method, filename), test_set, test_labels)
            for filename in listdir(path.join(CLASSIFIERS_DIR, method)) if filename.endswith(".pickle"))


def main():
    args = get_arguments("Classifier Benchmark")
    filterwarnings("ignore")
    print("Loading test set...")

    test_set, test_labels = load_object(path.join(TRAINING_SET_DIR, 'test_set.pickle'))

    print("Number of samples in test set:", len(test_labels))
    print(args.method)

    if args.method == MULTICLASS:
        bench_results = Parallel(n_jobs=args.n_jobs)(get_benchmark(MULTICLASS, test_set, test_labels))
    elif args.method == ENSEMBLE:
        bench_results = Parallel(n_jobs=args.n_jobs)(get_benchmark(ENSEMBLE, test_set, test_labels))
    elif args.method == 'all':
        bench_results = Parallel(n_jobs=args.n_jobs)(chain(get_benchmark(MULTICLASS, test_set, test_labels),
                                                           get_benchmark(ENSEMBLE, test_set, test_labels)))

    print("\n".join(sorted(bench_results)))
    print("Done.")


if __name__ == '__main__':
    main()
