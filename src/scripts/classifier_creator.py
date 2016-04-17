from itertools import chain
from os import path
from warnings import filterwarnings

from joblib import Parallel

from src.factories.classifier_factory import get_creator
from src.factories.learning_factory import LearningSetFactory
from src.settings import TRAINING_SET_DIR, MULTICLASS, ENSEMBLE
from src.utils.tools import save_object, get_arguments


def main():
    args = get_arguments("Classifier Creator")
    filterwarnings("ignore")

    print("Getting learning sets, using:", args.dataset.upper())

    train_set, train_labels, test_set, test_labels = \
        LearningSetFactory.get_learning_sets_and_labels(0.8, getattr(LearningSetFactory.DataSource, args.dataset))

    save_object(path.join(TRAINING_SET_DIR, "train_set.pickle"), [train_set, train_labels])
    save_object(path.join(TRAINING_SET_DIR, "test_set.pickle"), [test_set, test_labels])

    print("Done.")
    print("Creating classifiers...")

    if args.method == MULTICLASS:
        Parallel(n_jobs=args.n_jobs)(get_creator(MULTICLASS, train_set, train_labels))
    elif args.method == ENSEMBLE:
        Parallel(n_jobs=args.n_jobs)(get_creator(ENSEMBLE, train_set, train_labels))
    elif args.method == 'all':
        Parallel(n_jobs=args.n_jobs)(chain(get_creator(MULTICLASS, train_set, train_labels),
                                            get_creator(ENSEMBLE, train_set, train_labels)))

    print("Done.")

if __name__ == '__main__':
    main()
