from itertools import chain
from warnings import filterwarnings

from joblib import Parallel

from src.factories.classifier_factory import get_creator
from src.factories.learning_factory import LearningSetFactory
from src.settings import MULTICLASS, ENSEMBLE
from src.utils.tools import get_arguments


def main():
    print("Classifier Creator")
    args = get_arguments("Classifier Creator")
    filterwarnings("ignore")

    print("Getting learning sets, using:", args.dataset.upper())

    train_set, train_labels = \
        LearningSetFactory.get_full_learning_set_with_labels(getattr(LearningSetFactory.DataSource, args.dataset))

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
