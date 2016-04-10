from os import path, listdir

from joblib import Parallel, delayed

from src.factories import MulticlassClassifierFactory
from src.settings import CLASSIFIERS_DIR, TRAINING_SET_DIR
from src.utils import load_object


def benchmark_result(classifier_path, test_set, test_labels):
    classifier = MulticlassClassifierFactory.make_classifier_from_file(classifier_path)
    return "{0:25} : {1:.3f}".format(path.splitext(path.basename(classifier_path))[0],
                                     classifier.accuracy(test_set, test_labels))


def main():
    print("Loading test set...")

    test_set, test_labels = load_object(path.join(TRAINING_SET_DIR, 'test_set.pickle'))

    print("Benchmark")

    bench_results = Parallel(n_jobs=-1)(delayed(benchmark_result)
                                        (path.join(CLASSIFIERS_DIR, filename), test_set, test_labels)
                                        for filename in listdir(CLASSIFIERS_DIR) if filename.endswith(".pickle"))
    print("\n".join(sorted(bench_results)))
    print("Done.")


if __name__ == '__main__':
    main()
