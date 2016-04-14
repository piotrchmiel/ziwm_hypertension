from itertools import islice

import numpy as np
from joblib import dump, load
from openpyxl import load_workbook


class ExcelParser(object):

    def __init__(self, workbook_path, sheet_name):
        self.workbook = load_workbook(filename=workbook_path, read_only=True)
        self.worksheet = self.workbook[sheet_name]
        self.keys = []
        for row in islice(self.worksheet.rows, 0, 1):
            for cell in row:
                self.keys.append(cell.value)

    def get_rows(self):
        for row in islice(self.worksheet.rows, 1, None):
            feature_set = {}
            for key, cell in zip(self.keys, row):
                feature_set[key] = cell.value
                if key != "wy" and (isinstance(feature_set[key], int) or
                                    (isinstance(feature_set[key], str) and
                                     feature_set[key] != '?')):
                    feature_set[key] = float(feature_set[key])
                elif feature_set[key] == '?' or feature_set[key] is None:
                    feature_set[key] = np.nan
            yield feature_set


def save_object(file_location, class_object):
    dump(class_object, file_location, 3)


def load_object(file_location):
    return load(file_location)


def get_neighbors_above_threshold(y, neighbors, threshold):
    neighbors_list = []

    for neighbor in neighbors[0]:
        neighbors_list.append(y[neighbor])

    neighbors_count = len(neighbors_list)
    neighbors_set = set(neighbors_list)
    neighbors_set_tmp = neighbors_set.copy()
    for neighbor in neighbors_set_tmp:
        if not neighbors_list.count(neighbor) / neighbors_count > threshold:
            neighbors_set.remove(neighbor)

    return neighbors_set
