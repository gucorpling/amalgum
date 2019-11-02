import glob
import os
import re
import subprocess
from contextlib import contextmanager
import tempfile
import time
import traceback
from unicodedata import category

import requests as r
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from transformations.html import apply_html_transformations
from transformations.mwtext import apply_mwtext_transformations
from db.db import initialize as initialize_db, remove_db
from db import db

from lib.utils import Document
from lib.whitespace_tokenize import tokenize


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
GUM_DIR = os.path.dirname(FILE_DIR) + os.sep + "srcdata" + os.sep + "gum"
MIN_TOKEN_COUNT = 450
MAX_TOKEN_COUNT = 1100


# --------------------------------------------------------------------------------
# the meat
def convert(config, mwtext_object, dev_mode=False):
    mwtext = apply_mwtext_transformations(config, mwtext_object.text)

    # use Parsoid to produce HTML
    if config["parsoid_mode"] == "cli" or dev_mode:
        html = parsoid_convert_via_cli(config, mwtext)
    else:
        html = parsoid_convert_via_http(config, mwtext)
    html = apply_html_transformations(config, html, mwtext_object)
    return html


def parsoid_convert_via_cli(config, mwtext):
    command = ["parsoid/bin/parse.js", f"--domain={config['url']}"]
    if "api_url" in config:
        command.append(f"--apiURL={config['api_url']}")
    parser_subprocess = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    parser_subprocess.stdin.write(str(mwtext).encode("utf-8"))
    parser_subprocess.stdin.close()
    html = parser_subprocess.stdout.read().decode("utf-8")
    parser_subprocess.wait()
    return html


def parsoid_convert_via_http(config, mwtext):
    resp = r.post(
        f"http://localhost:8000/{config['family']}/v3/transform/wikitext/to/html/",
        {"wikitext": mwtext},
    )
    if resp.status_code != 200:
        raise Exception(f"Non-200 status code {resp.status_code}: {resp.reason}")
    return resp.text


# --------------------------------------------------------------------------------
# scraper specific methods
def write_user_config(config):
    with open(FILE_DIR + os.sep + "user-config.py", "w") as f:
        f.write(
            f"""
family_files['wikihow'] = 'https://www.wikihow.com/api.php'
family = '{config['family']}'
userinterface_lang = 'en'
mylang = 'en'
usernames['wikinews']['en'] = 'AutoGumBot'
"""
        )


def write_parsoid_config(config):
    with open(os.sep.join([FILE_DIR, "parsoid", "config.yaml"]), "w") as f:
        f.write(
            f"""
worker_heartbeat_timeout: 300000

logging:
    level: info

services:
  - module: lib/index.js
    entrypoint: apiServiceWorker
    conf:
        mwApis:
        - uri: {'https://' + config['url'] + '/w/api.php' if 'api_url' not in config else config['api_url']}
          domain: '{config['family']}'  # optional
"""
        )


@contextmanager
def boot_parsoid(config):
    write_parsoid_config(config)
    try:
        p = subprocess.Popen(
            ["npm", "start"],
            cwd=FILE_DIR + os.sep + "parsoid",
            stdout=subprocess.DEVNULL,
        )
        print("Started parsoid, sleeping to let it init...")
        time.sleep(3)
        yield p
    finally:
        p.terminate()  # send sigterm, or ...
        p.kill()  # send sigkill


def write_output(mwtext_object, gum_tei, output_dir):
    filename = mwtext_object.file_safe_url + "_" + str(mwtext_object.rev_id) + ".xml"
    filepath = os.sep.join([output_dir, filename])
    with open(filepath, "w") as f:
        f.write(gum_tei)


def genre(output_dir):
    if output_dir.endswith(os.sep):
        return output_dir.split(os.sep)[-2]
    else:
        return output_dir.split(os.sep)[-1]


def write_output_with_document_class(mwtext_object, gum_tei, output_dir, doc_number):
    html = re.sub(r"^<text[^>]*>", "", gum_tei)
    html = re.sub(r"</text>$", "", html)

    d = Document(
        title=mwtext_object.title,
        text=html,
        url=mwtext_object.url,
        date_collected=time.time(),
        date_created=mwtext_object.created_at.split("T")[0],
        date_modified=mwtext_object.modified_at.split("T")[0],
        genre=genre(output_dir),
        docnum=doc_number,
    )
    d.short_title = d.make_short_title()
    d.serialize(out_dir=output_dir)


def rough_word_count(gum_tei):
    tokens = tokenize(gum_tei).split("\n")
    tokens = [
        t
        for t in tokens
        if (
            not t.startswith("<")
            and not t.isspace()
            and any(category(c).startswith("L") for c in t)
        )
    ]
    return len(tokens)


def process_page(config, page, output_dir, doc_number):
    print(f"Processing `{str(page)}`... ", end="")
    mwtext_object = get_mwtext_object(page)
    gum_tei = convert(config, mwtext_object)
    token_count = rough_word_count(gum_tei)
    if MIN_TOKEN_COUNT <= token_count <= MAX_TOKEN_COUNT:
        write_output_with_document_class(mwtext_object, gum_tei, output_dir, doc_number)
        print("done.")
        return token_count
    else:
        print()
        print(
            f'\tSKIPPING: "{page.title(as_url=True)} has roughly {token_count} tokens.'
        )
        return 0


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
        config["page_generation"]["endpoint"],
        site=site,
        **config["page_generation"]["params"],
    )
    return lg


def urls_already_scraped_for_genre(genre):
    files = glob.glob(GUM_DIR + os.sep + "xml" + os.sep + f"GUM_{genre}*")
    urls = []
    for file in files:
        with open(file, "r") as f:
            match = re.search(r'sourceURL="([^"]*)"', f.read())
        if match:
            urls.append(match.groups()[0].replace("-", "_"))
        else:
            raise Exception(f"Couldn't find a URL for {file}.")
    return urls


def already_scraped(urls, page):
    page_url = page.title(as_url=True)
    return any(url.endswith(page_url) for url in urls)


def scrape(config_filepath, output_dir):
    config = load_config(config_filepath)

    # write pywikibot config
    pywikibot = init_pywikibot(config)
    site = pywikibot.Site()

    # initialize db and output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    initialize_db(output_dir)

    urls_already_scraped = urls_already_scraped_for_genre(genre(output_dir))

    i = 0
    word_count_total = 0
    with boot_parsoid(config) as _:
        for page_dict in page_generator(config, pywikibot, site):
            try:
                page = pywikibot.Page(site, page_dict["title"])
                if already_scraped(urls_already_scraped, page):
                    print(
                        f'\tSKIPPING: "{page.title(as_url=True)}" has already been included in GUM.'
                    )
                else:
                    word_count_total += process_page(config, page, output_dir, i)
                    if word_count_total > 0:
                        i += 1
            except Exception as e:
                print("Oops! Something went wrong.")
                traceback.print_exc()

            if "rate_limit" in config:
                time.sleep(config["rate_limit"])
            else:
                time.sleep(1)

    print("Finished processing. Total word count: ", word_count_total)


def convert_specific_article(config_filepath, url):
    config = load_config(config_filepath)
    pywikibot = init_pywikibot(config)
    monkey_patch_pywikibot(pywikibot)
    site = pywikibot.Site()
    page = pywikibot.Page(site, url[url.rfind("/") + 1 :])
    remove_db(tempfile.gettempdir())
    initialize_db(tempfile.gettempdir())
    mwtext_object = get_mwtext_object(page, dev_mode=True)
    return convert(config, mwtext_object, dev_mode=True)


def monkey_patch_pywikibot(pywikibot):
    old_sametitle = pywikibot.BaseSite.sametitle

    def new_sametitle(self, title1, title2):
        title1 = title1.replace("-", "_")
        title2 = title2.replace("-", "_")
        return old_sametitle(self, title1, title2)

    pywikibot.BaseSite.sametitle = new_sametitle


# ------------------------------------------------------------------------------
# utils
def get_mwtext_object(page, dev_mode=False):
    if dev_mode or not db.mwtext_exists(str(page)):
        title = page.title()
        url = page.full_url()
        file_safe_url = page.title(as_filename=True)
        latest_revision = page.latest_revision
        rev_id = str(latest_revision["revid"])
        text = latest_revision["text"]
        oldest_revision_time = page.oldest_revision.timestamp.isoformat()
        latest_revision_time = latest_revision.timestamp.isoformat()
        db.add_text(
            str(page),
            url,
            rev_id,
            text,
            title,
            file_safe_url,
            oldest_revision_time,
            latest_revision_time,
        )
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
    p.add_argument(
        "--output-dir",
        default=os.sep.join([FILE_DIR, "output", "wikinews"]),
        help="directory that will contain the output files",
    )
    args = p.parse_args()

    if args.method == "scrape":
        scrape(args.config, args.output_dir)
    elif args.method == "url":
        assert args.url
        print(convert_specific_article(args.config, args.url))
    else:
        raise Exception(f"Unknown method '{args.method}'")
