import glob
import sys
import re
import os

NUMBER_PATTERN = re.compile(r"(\d+).xml")


def pad(numstring, padding_length=4):
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
            new_i = pad(str(new_i))
            old_i = str(old_i)
            if new_i != old_i:
                new_name = new_filename(filename, old_i, new_i)
                print(f"Renaming:\t'{filename}' -> '{new_name}'")
                os.rename(filename, new_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            """usage:

    python token_count.py xml_dir...
"""
        )
        sys.exit(0)
    main(sys.argv[1:])
