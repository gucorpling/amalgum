import glob
import sys
import re
import os
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from bs4 import BeautifulSoup

from lib.whitespace_tokenize import tokenize

NUMBER_PATTERN = re.compile(r"(\d+).xml")
MIN_WORD_COUNT = 450
MAX_WORD_COUNT = 1100


def rough_word_count(gum_tei):
    tokens = tokenize(gum_tei).split("\n")
    tokens = [t for t in tokens if (not t.startswith("<") and not t.isspace())]
    return len(tokens)


class GUMTEIFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split(os.sep)[-1][:-4]
        self.doc_number = int(re.search(NUMBER_PATTERN, self.filepath).group(1))
        with open(filepath, "r") as f:
            self.text = f.read()
        self.soup = BeautifulSoup(self.text, "html.parser")

        # get <text> attributes
        for attr_name, attr_value in self.soup.contents[0].attrs.items():
            setattr(self, attr_name, attr_value)

        self._text_length = None

    @property
    def text_length(self):
        if self._text_length is None:
            self._text_length = rough_word_count(self.text)
        return self._text_length

    def __str__(self):
        return self.filepath

    def __repr__(self):
        return self.filepath


def filename_and_id_identical(files):
    for f in files:
        if not f.filename == f.id:
            print(f"{f}: has filename {f.filename}, but id is {f.id}")


def file_numbers_consecutive(files):
    last_number = -1
    for f in sorted(files, key=lambda x: x.doc_number):
        if f.doc_number != last_number + 1:
            print(
                f"{f}: document number is {f.doc_number}, but the previous file's number was {last_number}"
            )
        last_number = f.doc_number


def word_count_within_bounds(files):
    for f in files:
        if not MIN_WORD_COUNT <= f.text_length <= MAX_WORD_COUNT:
            print(
                f"{f}: has {f.text_length} words, which is not between {MIN_WORD_COUNT} and {MAX_WORD_COUNT}"
            )


def valid_xml(files):
    for f in files:
        try:
            ET.fromstring(f.text)
        except ParseError as e:
            print(f"{f}: has invalid XML")
            print(str(e))


def main(directory):
    files = [
        GUMTEIFile(filepath) for filepath in glob.glob(directory + os.sep + "*.xml")
    ]
    filename_and_id_identical(files)
    file_numbers_consecutive(files)
    word_count_within_bounds(files)
    valid_xml(files)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            """usage:

    python token_count.py xml_dir
"""
        )
        sys.exit(0)
    main(sys.argv[1])
