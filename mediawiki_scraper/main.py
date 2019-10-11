import os
import subprocess

import requests as r
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from transformations.html import apply_html_transformations, fix_root
from transformations.mwtext import apply_mwtext_transformations
import db.db as db


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
GUM_TEI_DIR = "gum_tei"


# --------------------------------------------------------------------------------
# the meat
def convert(config, mwtext_object):
    mwtext = apply_mwtext_transformations(config, mwtext_object.text)

    # use Parsoid to produce HTML
    if config['parsoid_mode'] == "http":
        html = parsoid_convert_via_http(mwtext)
    else:
        html = parsoid_convert_via_cli(config, mwtext)
    html = apply_html_transformations(config, html, mwtext_object)
    return html


def parsoid_convert_via_cli(config, mwtext):
    parser_subprocess = subprocess.Popen(
        ["parsoid/bin/parse.js", f"--domain={config['url']}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    parser_subprocess.stdin.write(str(mwtext).encode("utf-8"))
    parser_subprocess.stdin.close()
    html = parser_subprocess.stdout.read().decode("utf-8")
    parser_subprocess.wait()
    return html


def parsoid_convert_via_http(mwtext):
    resp = r.post(
        "http://localhost:8000/wikinews/v3/transform/wikitext/to/html/",
        {"wikitext": mwtext}
    )
    if resp.status_code != 200:
        raise Exception("Non-200 status code: " + resp)
    return resp.text

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


def write_output(mwtext_object, gum_tei):
    filename = mwtext_object.file_safe_url + "_" + str(mwtext_object.rev_id) + ".xml"
    filepath = os.sep.join([GUM_TEI_DIR, filename])
    with open(filepath, "w") as f:
        f.write(gum_tei)


def process_page(config, page):
    print(f"Processing `{str(page)}`... ", end="")
    mwtext_object = get_mwtext_object(page)
    gum_tei = convert(config, mwtext_object)
    write_output(mwtext_object, gum_tei)
    print("done.")


# ------------------------------------------------------------------------------
# different methods for acquiring text

def init_pywikibot(config):
    # pywikibot requires a file named user-config.py to be initialized before import
    # let's humor it...
    write_user_config(config)
    import pywikibot
    return pywikibot


def page_generator(config, pywikibot, site):
    from pywikibot.data.api import ListGenerator
    lg = ListGenerator(
        config['page_generation']['endpoint'],
        site=site,
        **config['page_generation']['params']
    )
    return lg


def scrape(config_filepath):
    config = load_config(config_filepath)
    pywikibot = init_pywikibot(config)
    site = pywikibot.Site()
    if not os.path.exists(GUM_TEI_DIR):
        os.mkdir(GUM_TEI_DIR)

    for page_dict in page_generator(config, pywikibot, site):
        try:
            page = pywikibot.Page(site, page_dict["title"])
            process_page(config, page)
        except Exception as e:
            print("Oops! Something went wrong.")
            print(e)


def convert_specific_article(config_filepath, url):
    config = load_config(config_filepath)
    pywikibot = init_pywikibot(config)
    site = pywikibot.Site()
    page = pywikibot.Page(site, url[url.rfind("/")+1:])
    mwtext_object = get_mwtext_object(page)
    return convert(config, mwtext_object)


# ------------------------------------------------------------------------------
# utils
def get_mwtext_object(page):
    if not db.mwtext_exists(str(page)):
        title = page.title()
        url = page.full_url()
        file_safe_url = page.title(as_filename=True)
        latest_revision = page.latest_revision
        rev_id = str(latest_revision["revid"])
        text = latest_revision["text"]
        db.add_text(str(page), url, rev_id, text, title, file_safe_url)
    return db.get_mwtext(str(page))


def load_config(config_filepath):
    with open(config_filepath, "r") as f:
        return yaml.load(f.read(), Loader=Loader)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        help="How to obtain the mwtext; one of: 'scrape', 'url'",  # future: dump?
        default="scrape",
    )
    p.add_argument("--url", help="Url location for method 'url'")
    p.add_argument(
        "--config",
        default=os.sep.join([FILE_DIR, "configs", "wikinews.yaml"]),
        help="yaml config that describes the MediaWiki instance to be scraped",
    )
    args = p.parse_args()

    if args.method == "scrape":
        scrape(args.config)
    elif args.method == "url":
        assert args.url
        print(convert_specific_article(args.config, args.url))
    else:
        raise Exception(f"Unknown method '{args.method}'")
