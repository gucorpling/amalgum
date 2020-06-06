import glob
import os
import sys

from tqdm import tqdm
from lib.whitespace_tokenize import tokenize


def rough_word_count(gum_tei):
    tokens = tokenize(gum_tei).split("\n")
    tokens = [t for t in tokens if (not t.startswith("<") and not t.isspace())]
    return len(tokens)


def main(d):

    for dirname in d:
        file_names = glob.glob(os.path.join(dirname, "*.xml"))
        total_wc = 0
        for file_name in tqdm(file_names):
            with open(file_name, "r") as f:
                total_wc += rough_word_count(f.read())

        print(f"Total word count for {os.path.join(dirname, '*.xml')}:\t{total_wc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            """usage:

    python token_count.py xml_dir...
"""
        )
        sys.exit(0)
    main(sys.argv[1:])
