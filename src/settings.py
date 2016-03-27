from os import path

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

BASE_DIR = path.dirname(path.dirname(__file__))

TRAINING_SET_DIR = path.join(BASE_DIR, "Training set")
TRAINING_SET_FILENAME = "hyper.xlsx"
SHEET_NAME = "hyper"
CLASSIFIERS_DIR = path.join(BASE_DIR, "Classifiers")
ALGORITHMS = {'ada_boost_classifier': AdaBoostClassifier, 'random_forest': RandomForestClassifier,
              'bagging_classifier': BaggingClassifier}