from itertools import chain
from warnings import filterwarnings

from src.factories.learning_factory import LearningSetFactory
from src.settings import METHODS, MULTICLASS, ENSEMBLE
from src.utils.helpers import cross_val_score
from src.utils.tools import get_arguments


def main():
    args = get_arguments("Cross Validation")
    filterwarnings("ignore")
    print("Start Cross Validation")

    train_set, train_labels = LearningSetFactory.get_full_learning_set_with_labels(
        LearningSetFactory.DataSource.hypertension)

    print("Number of samples in learning set:", len(train_labels))
    if args.method == MULTICLASS:
        print(cross_val_score(METHODS[MULTICLASS], train_set, train_labels, n_jobs=args.n_jobs, cv=10))
    elif args.method == ENSEMBLE:
        print(cross_val_score(METHODS[ENSEMBLE], train_set, train_labels, n_jobs=args.n_jobs, cv=10))
    elif args.method == 'all':
        print(cross_val_score(chain(METHODS[MULTICLASS], METHODS[ENSEMBLE]),
                              train_set, train_labels, n_jobs=args.n_jobs, cv=10))

if __name__ == '__main__':
    main()

