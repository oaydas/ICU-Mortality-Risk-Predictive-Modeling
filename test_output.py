import sys
import csv
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input data file. Must be in csv.")


def main() -> None:
    """Check the validity of challenge submission."""
    args = parser.parse_args()
    input_file = Path(args.input)

    if input_file.suffix != ".csv":
        raise RuntimeError("File is not of .csv format.")

    with open(input_file, newline="") as csvfile:
        filereader = csv.reader(csvfile)
        i = 0
        for row in filereader:
            if i == 0:
                if row[0] != "label":
                    raise RuntimeError("File does not have the correct column name.")
            else:
                rating = int(row[0])
                if rating != -1 and rating != 0 and rating != 1:
                    raise RuntimeError("INVALID VALUE: values need to be -1, 0, or 1.")
            i += 1
        if i != 2001:
            raise RuntimeError("File does not have exactly 2001 rows.")
        
        print("SUCCESS: csv file is valid.")


if __name__ == "__main__":
    main()
