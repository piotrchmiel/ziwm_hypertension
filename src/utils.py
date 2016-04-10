from itertools import islice

import numpy as np
from joblib import dump, load
from openpyxl import load_workbook


class ExcelParser(object):

    def __init__(self, workbook_path, sheet_name):
        self.wb = load_workbook(filename=workbook_path, read_only=True)
        self.ws = self.wb[sheet_name]
        self.keys = []
        for row in islice(self.ws.rows, 0, 1):
            for cell in row:
                self.keys.append(cell.value)

    def get_rows(self):
        for row in islice(self.ws.rows, 1, None):
            feature_set = {}
            for key, cell in zip(self.keys, row):
                feature_set[key] = cell.value
                if key != "wy" and (type(feature_set[key]) is int or \
                        (type(feature_set[key]) is str and feature_set[key] != '?')):
                    feature_set[key] = float(feature_set[key])
                elif feature_set[key] == '?' or feature_set[key] is None:
                    feature_set[key] = np.nan
            yield feature_set


def save_object(file_location, object):
        dump(object, file_location, 3)


def load_object(file_location):
    return load(file_location)
