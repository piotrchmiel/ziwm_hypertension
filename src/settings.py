from os import path

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.dynamic_OvO import DynamicOneVsOneClassifier
from src.dynamic_OvR import DynamicOneVsRestClassifier

BASE_DIR = path.dirname(path.dirname(__file__))

ENSEMBLE = 'ensemble'
MULTICLASS = 'multiclass'

TRAINING_SET_DIR = path.join(BASE_DIR, "Training set")
TRAINING_SET_FILENAME = "hyper.xlsx"
SHEET_NAME = "hyper"

CLASSIFIERS_DIR = path.join(BASE_DIR, "Classifiers")

DEFAULT_PARAMETERS = {ENSEMBLE: {'n_estimators': 50},
                      MULTICLASS: {'n_jobs': -1, 'estimator': DecisionTreeClassifier()}}

METHODS = {ENSEMBLE: {'ada_boost_classifier': (AdaBoostClassifier, DEFAULT_PARAMETERS[ENSEMBLE]),
                      'random_forest': (RandomForestClassifier, DEFAULT_PARAMETERS[ENSEMBLE]),
                      'bagging_classifier': (BaggingClassifier, DEFAULT_PARAMETERS[ENSEMBLE])},
           MULTICLASS: {'one_vs_one': (OneVsOneClassifier, DEFAULT_PARAMETERS[MULTICLASS]),
                        'dynamic_one_vs_one': (DynamicOneVsOneClassifier, DEFAULT_PARAMETERS[MULTICLASS]),
                        'one_vs_rest': (OneVsRestClassifier, DEFAULT_PARAMETERS[MULTICLASS]),
                        'dynamic_one_vs_rest': (DynamicOneVsRestClassifier, DEFAULT_PARAMETERS[MULTICLASS])}}
