from csv import reader
from itertools import islice


class CsvParser(object):
    def __init__(self, filename):
        with open(filename) as f:
            csv_reader = reader(f, delimiter=';')
            records = list(csv_reader)
        self.records = records
        self.keys = []
        for row in islice(self.records, 0, 1):
            for cell in row:
                self.keys.append(cell)

    def get_rows(self):
        for row in islice(self.records, 1, None):
            feature_set = {}
            for key, cell in zip(self.keys, row):
                feature_set[key] = cell
                if isinstance(feature_set[key], str):
                    try:
                        feature_set[key] = float(feature_set[key])
                    except ValueError:
                        pass
            yield feature_set

    def get_keys(self):
        return self.keys
