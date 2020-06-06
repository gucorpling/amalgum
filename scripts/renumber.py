import glob
import sys
import re
import os

NUMBER_PATTERN = re.compile(r"(\d+).xml")
ID_PATTERN = re.compile(r' id="([^"]*)"')


def pad(numstring, padding_length=3):
    return ("0" * (padding_length - len(str(numstring)))) + str(numstring)


def new_filepath(filepath, old_i, new_i):
    assert filepath.endswith(old_i + ".xml")
    filepath = filepath[: len(filepath) - len(old_i + ".xml")]
    filepath = filepath + new_i + ".xml"
    return filepath


def main(d):
    for dirname in d:
        filepaths_with_numbers = []

        filepaths = glob.glob(f"{dirname}/*.xml")
        for filepath in filepaths:
            match = re.search(NUMBER_PATTERN, filepath.split(os.sep)[-1])
            if not match:
                print(f"Couldn't find a number in {filepath}!")
                sys.exit(1)
            else:
                filepaths_with_numbers.append((match.group(1), filepath))

        for new_i, (old_i, filepath) in enumerate(sorted(filepaths_with_numbers, key=lambda x: int(x[0]))):
            new_i = pad(str(new_i))
            old_i = str(old_i)
            if new_i != old_i:
                new_name = new_filepath(filepath, old_i, new_i)
                print(f"Renaming:\t'{filepath}' -> '{new_name}'")
                os.rename(filepath, new_name)

                filepath = new_name

            with open(filepath, "r") as f:
                s = f.read()

            filename = filepath.split(os.sep)[-1][:-4]
            match = re.search(ID_PATTERN, s)
            if match is None:
                print(f"{filepath} doesn't have an id attribute!")
                sys.exit(1)
            elif match.group(1) != filename:
                print(
                    f"<text>'s id attribute is {match.group(1)}, but filename is {filename}"
                )
                s = s.replace(f' id="{match.group(1)}"', f' id="{filename}"')
                print("\tReplaced id attribute with the filepath.")
                with open(filepath, "w") as f:
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
