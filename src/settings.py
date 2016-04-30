from os import path

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.classifiers.dynamic_OvO import DynamicOneVsOneClassifier
from src.classifiers.dynamic_OvR import DynamicOneVsRestClassifier

BASE_DIR = path.dirname(path.dirname(__file__))

ENSEMBLE = 'ensemble'
MULTICLASS = 'multiclass'

TRAINING_SET_DIR = path.join(BASE_DIR, "Training set")
TIME_BENCH_DIR = path.join(BASE_DIR, "Time Benchmark")
HYPER_TRAINING_SET = "hyper.xlsx"
HYPER_SHEET_NAME = "hyper"
ISOLET_TRAINING_SET = "isolet.csv"
AUSLAN_TRAINING_SET = "auslan.csv"
KDDCUP_TRAINING_SET = "kddcup.csv"
STUDENT_ALCOHOL_TRAINING_SET = "student_alcohol.csv"
ADULT_TRAINING_SET = "adult.csv"
WINE_QUALITY_TRAINING_SET = "wine_quality.csv"

CLASSIFIERS_DIR = path.join(BASE_DIR, "Classifiers")

DEFAULT_PARAMETERS = {ENSEMBLE: {'n_estimators': 50},
                      MULTICLASS: {'n_jobs': -1, 'estimator': DecisionTreeClassifier()}}

METHODS = {ENSEMBLE: {'ada_boost_classifier': (AdaBoostClassifier, DEFAULT_PARAMETERS[ENSEMBLE]),
                      'random_forest': (RandomForestClassifier, DEFAULT_PARAMETERS[ENSEMBLE]),
                      'bagging_classifier': (BaggingClassifier, DEFAULT_PARAMETERS[ENSEMBLE])},
           MULTICLASS: {'one_vs_one': (OneVsOneClassifier, DEFAULT_PARAMETERS[MULTICLASS]),
                        'one_vs_rest': (OneVsRestClassifier, DEFAULT_PARAMETERS[MULTICLASS]),
                        'dynamic_one_vs_one_all': (DynamicOneVsOneClassifier, dict(
                            {'threshold': 0.0, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        'dynamic_one_vs_rest_all': (DynamicOneVsRestClassifier, dict(
                            {'threshold': 0.0, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        'dynamic_one_vs_one_5': (DynamicOneVsOneClassifier, dict(
                            {'threshold': 0.05, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        'dynamic_one_vs_rest_5': (DynamicOneVsRestClassifier, dict(
                            {'threshold': 0.05, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        'dynamic_one_vs_one_10': (DynamicOneVsOneClassifier, dict(
                            {'threshold': 0.1, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        'dynamic_one_vs_rest_10': (DynamicOneVsRestClassifier, dict(
                            {'threshold': 0.1, 'n_neighbors': 18}, **DEFAULT_PARAMETERS[MULTICLASS])),
                        }}
