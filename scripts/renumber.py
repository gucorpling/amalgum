import glob
import sys
import re
import os
from bs4 import BeautifulSoup

NUMBER_PATTERN = re.compile(r"(\d+).xml")
ID_PATTERN = re.compile(r' id="([^"]*)"')


def pad(numstring, padding_length=3):
    return ("0" * (padding_length - len(str(numstring)))) + str(numstring)


def new_filename(filename, old_i, new_i):
    assert filename.endswith(old_i + ".xml")
    filename = filename[: len(filename) - len(old_i + ".xml")]
    filename = filename + new_i + ".xml"
    return filename


def main(d):
    for dirname in d:
        filenames_with_numbers = []

        filenames = glob.glob(f"{dirname}/*.xml")
        for filename in filenames:
            match = re.search(NUMBER_PATTERN, filename.split(os.sep)[-1])
            if not match:
                print(f"Couldn't find a number in {filename}!")
                sys.exit(1)
            else:
                filenames_with_numbers.append((int(match.group(1)), filename))

        for new_i, (old_i, filename) in enumerate(
            sorted(filenames_with_numbers, key=lambda x: x[0])
        ):
            new_i = str(new_i)
            old_i = str(old_i)
            if new_i != old_i:
                new_name = new_filename(filename, old_i, new_i)
                print(f"Renaming:\t'{filename}' -> '{new_name}'")
                os.rename(filename, new_name)

                filename = new_name

            with open(filename, "r") as f:
                s = f.read()

            match = re.search(ID_PATTERN, s)
            if match is None:
                print(f"{filename} doesn't have an id attribute!")
                sys.exit(1)
            elif match.group(1) != filename[:-4]:
                print(
                    f"<text>'s id attribute is {match.group(1)}, but filename is {filename[:-4]}"
                )
                s = s.replace(f' id="{match.group(1)}"', f' id="{filename[:-4]}"')
                print("\tReplaced id attribute with the filename.")
                with open(filename, "w") as f:
                    f.write(s)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            """usage:

    python token_count.py xml_dir...
"""
        )
        sys.exit(0)
    main(sys.argv[1:])
