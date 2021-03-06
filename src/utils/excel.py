from itertools import islice

import numpy as np
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
                if key != self.keys[-1] and (isinstance(feature_set[key], int) or
                                    (isinstance(feature_set[key], str) and
                                     feature_set[key] != '?')):
                    feature_set[key] = float(feature_set[key])
                elif feature_set[key] == '?' or feature_set[key] is None:
                    feature_set[key] = np.nan
            yield feature_set
