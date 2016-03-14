from collections import defaultdict
from os import path
from pprint import pprint
from src.utils import ExcelParser
from src.settings import TRAINING_SET_DIR, TRAINING_SET_FILENAME, SHEET_NAME


def main():
    parser = ExcelParser(path.join(TRAINING_SET_DIR, TRAINING_SET_FILENAME), SHEET_NAME)

    rows = defaultdict(list)
    for row in parser.get_rows():
        rows[row['wy']].append(row)

    pprint(rows)

if __name__ == '__main__':
    main()
