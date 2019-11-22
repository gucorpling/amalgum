import glob
import sys


def main(d):
    for dirname in d:
        filepaths = glob.glob(f"{dirname}/*.xml")
        for filepath in filepaths:
            with open(filepath, "r") as f:
                s = f.read()
            if s.find("shortTile=") == -1:
                print(f"{filepath} doesn't have short title!")
                sys.exit(1)
            else:
                # not sure what format/chars everyone used for their shortTitle, so the safest way not using regex:
                start = s.find('shortTile="') + len('shortTile="')
                end = s.find('"', start)
                sub = s[start:end]
                if sub.find("--") != -1:
                    s = s[:start] + sub.replace("--", "-") + s[end:]
                    with open(filepath, "w") as f:
                        f.write(s)
                        print(f"{filepath} modified.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            """usage:

    python fix_short_title.py xml_dir...
"""
        )
    main(sys.argv[1:])
