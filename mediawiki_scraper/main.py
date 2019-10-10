import os
import glob
import subprocess

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from slugify import slugify

from transformations.html import apply_html_transformations, fix_root
from transformations.mwtext import apply_mwtext_transformations


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MWTEXT_DIR = "mwtext"
GUM_TEI_DIR = "gum_tei"


# --------------------------------------------------------------------------------
# the meat
def convert(config, mwtext, revid, url):
    mwtext = apply_mwtext_transformations(config, mwtext)

    # use Parsoid to produce HTML
    parser_subprocess = subprocess.Popen(
        ["parsoid/bin/parse.js", f"--domain={config['url']}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    parser_subprocess.stdin.write(str(mwtext).encode("utf-8"))
    parser_subprocess.stdin.close()
    html = parser_subprocess.stdout.read().decode("utf-8")
    parser_subprocess.wait()

    html = apply_html_transformations(config, html, revid, slugify(url), url)
    return html


# --------------------------------------------------------------------------------
# scraper specific methods
def write_user_config(config):
    with open(FILE_DIR + os.sep + "user-config.py", "w") as f:
        f.write(
            f"""
family = '{config['family']}'
userinterface_lang = 'en'
mylang = 'en'
usernames['wikinews']['en'] = 'AutoGumBot'
"""
        )


def page_to_mwtext_filepath(url, revid):
    return os.sep.join([MWTEXT_DIR, slugify(url) + "_" + revid])


def check_cache(url):
    return glob.glob(MWTEXT_DIR + os.sep + slugify(url) + "_*")


def read_page_from_cache(hit):
    with open(hit, "r") as f:
        return f.read()


def write_page_to_cache(url, revid, mwtext):
    with open(page_to_mwtext_filepath(url, revid), "w") as f:
        f.write(mwtext)


def page_to_gum_tei_filepath(url, revid):
    return os.sep.join([GUM_TEI_DIR, slugify(url) + "_" + revid + ".xml"])


def write_output(url, revid, gum_tei):
    filepath = page_to_gum_tei_filepath(url, revid)
    with open(filepath, "w") as f:
        f.write(gum_tei)
    print(f"    Output written to {filepath}")


def process_page(config, page):
    url = page.full_url()
    print(f"Processing `{url}`...")

    cache_hits = check_cache(url)
    if cache_hits:
        hit = cache_hits[0]
        revid = hit.split("_")[-1]
        mwtext = read_page_from_cache(hit)
        print(f"    (Found in cache.)")
    else:
        latest_revision = page.latest_revision
        revid = str(latest_revision["revid"])
        mwtext = latest_revision["text"]
        write_page_to_cache(url, revid, mwtext)

    gum_tei = convert(config, mwtext, revid, url)
    write_output(url, revid, gum_tei)


# ------------------------------------------------------------------------------
# different methods for acquiring text


def scrape(config_filepath):
    config = load_config(config_filepath)

    # pywikibot requires a file named user-config.py to be initialized before import
    # let's humor it...
    write_user_config(config)
    import pywikibot

    site = pywikibot.Site()

    if not os.path.exists(MWTEXT_DIR):
        os.mkdir(MWTEXT_DIR)
    if not os.path.exists(GUM_TEI_DIR):
        os.mkdir(GUM_TEI_DIR)

    for page in site.allpages():
        process_page(config, page)


def convert_file(config_filepath, mwtext_filepath):
    with open(mwtext_filepath, "r") as f:
        doc = f.read()
    config = load_config(config_filepath)
    return convert(config, doc, "DEBUG", "DEBUG")


# ------------------------------------------------------------------------------
# utils
def load_config(config_filepath):
    with open(config_filepath, "r") as f:
        return yaml.load(f.read(), Loader=Loader)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        help="How to obtain the mwtext; one of: 'scrape', 'file'",  # future: dump?
        default="scrape",
    )
    p.add_argument("--file", help="File location for method 'file'")
    p.add_argument(
        "--config",
        default=os.sep.join([FILE_DIR, "configs", "wikinews.yaml"]),
        help="yaml config that describes the MediaWiki instance to be scraped",
    )
    args = p.parse_args()

    if args.method == "scrape":
        scrape(args.config)
    elif args.method == "file":
        assert args.file
        print(convert_file(args.config, args.file))
    else:
        raise Exception(f"Unknown method '{args.method}'")
