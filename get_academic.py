import urllib.request, bz2, io, os, sys, re, shutil, random

# from dateparser.search import search_dates  # This one is worse than datefinder
from collections import OrderedDict
from multiprocessing import cpu_count
from glob import glob
from random import shuffle, seed
import requests
from time import sleep

try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup
from lib.utils import Document

GENRE_LIMIT = 50000
PASSAGE_LIMIT = 850
PASSAGE_THREASH = 400

seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
RAW_PATH = script_dir + os.sep + "data" + os.sep + "academic"


class DocumentWithRaw(Document):
    def __init__(self, genre="academic"):
        super().__init__(genre)
        self.raw_text = ""


def write_file(path, doc):
    text = doc.raw_text
    title = doc.short_title
    PATH = path + os.sep
    if os.path.exists(PATH + f"{title}.txt"):
        with open(PATH + f"{title}-1.txt", "w", encoding="utf8") as f:
            f.write(text)
    else:
        with open(PATH + f"{title}.txt", "w", encoding="utf8") as f:
            f.write(text)


def soup(url):
    html = requests.get(url).text
    parseed = BeautifulSoup(html)
    return parseed


def clean_text(text):
    # split the text by sections, randomly select one section
    if text.startswith('<div class="html-p"'):
        text = text
    else:
        index = len(re.findall('data-nested="1"', text)) - 1
        if index > 0:
            index = random.randint(0, index)
        sep = re.split(
            '<section id=".{1,30}" type=".{1,30}"?><h2 data-nested="1">', text
        )[1:]
        if len(sep) <= index:
            return ""
        text = re.split(
            '<section id=".{1,30}" type=".{1,30}"?><h2 data-nested="1">', text
        )[1:][index]

    text = re.sub(
        '<a href=(".*?").*?>(.*?)</a>', "<ref target=\g<1>>\g<2></ref>", text
    )  # href
    text = re.sub(
        '<a class="html-fig".*?>(.*?)</a>', "<figure>\g<1></figure>", text
    )  # figure
    text = re.sub(" \[<a.*?\]", "", text)  # reference
    text = re.sub("</?a.*?>", "", text)
    text = re.sub("</?su[b|p]>", "", text)  # special characters
    text = re.sub("<math[\w\W]*?/math>", "", text)  # math

    text = re.sub("(</?h).*?>", "\g<1>ead>", text)  # head
    text = re.sub("(</head>)(<head>)", "\g<1>\n\g<2>", text)  # head
    text = re.sub("</?section.*?>", "", text)  # section

    text = re.sub('<div class="html-p"><ul class=".*?">', "<list>\n", text)  # list head
    text = re.sub("</ul></div>", "</list>\n", text)  # list tail
    text = re.sub('<li><div class="html-p">', "<item>", text)  # item head
    text = re.sub("</div></li>", "</item>\n", text)  # item tail

    text = re.sub('</div><div class="html-p">', " </p> \n<p>", text)  # paragraph
    text = re.sub("</div>(<head>)", " </p> \n\g<1>", text)  # paragraph
    text = re.sub('<div class="html-p">', "\n<p>\n", text)  # paragraph

    text = re.sub(
        '<span class="html-italic">(.*?)</span>', '<hi rend="italic">\g<1></hi>', text
    )  # italic
    text = re.sub("</?(di|a|m|span|label).*?>", "", text)  # clean-up
    text = re.sub("\n\n\n+", "\n\n", text)  # clean-up
    text = re.sub("\n ", "\n", text)  # clean-up
    text = "<head>" + text
    text += " </p>"

    # truncate
    passage = ""
    for p in text.split("</p>")[:-1]:
        if passage.count(" ") + p.count(" ") <= PASSAGE_LIMIT:
            passage = passage + p + "</p>\n"
        else:
            break
    return passage


def get_texts(url):
    count = 0
    TEXT_LEN = 0
    prev_articles = []

    def get_url(def_url, regex, find):
        try:
            parsed_html = soup(def_url)
        except:
            sleep(5)
            parsed_html = soup(def_url)
        def_html = parsed_html.find_all(find[0], find[1])
        def_href = re.findall(regex, str(def_html))
        def_url = [url + href for href in def_href]
        return def_url

    # Get full text of an article if it has html format to view
    def get_fulltext(subject, article_url):
        parsed_html = soup(article_url)
        article_html = parsed_html.find(id="html_link")
        if article_html:
            current_doc = DocumentWithRaw(genre="academic")
            parsed_html = soup(article_url + "/htm")

            title = re.search(
                '<div id="html-article-title">(.*)</div>',
                str(parsed_html.find_all(id="html-article-title")),
            )
            if title:
                title = title.group(1)
                title = re.sub("</?.*?>", "", title)
                current_doc.title = title
                # current_doc.subject = subject
                current_doc.author = ", ".join(
                    re.findall(
                        '<meta content="(.*?)" name=".*?creator"/>', str(parsed_html)
                    )
                )
                current_doc.date = re.search(
                    '<meta content="(.*?)" name=".*?date"/>', str(parsed_html)
                ).group(1)
                # current_doc.publisher = re.search("<meta content=\"(.*?)\" name=\".*?publisher\"/>", str(parsed_html)).group(1)
                raw_text = re.search(
                    '<div class="html-body">\n([\w\W]*)</div>',
                    str(parsed_html.findAll("div", {"class": "html-body"})),
                ).group(1)
                current_doc.raw_text = BeautifulSoup(raw_text, "lxml").text
                current_doc.text = clean_text(raw_text)
                if current_doc.text.count(" ") < PASSAGE_THREASH:
                    return None
                current_doc.docnum = count
                current_doc.url = article_url + "/htm"
                return current_doc
            else:
                return None
        else:
            return None

    sub_urls = get_url(
        url,
        'href="(/subject.*)"',
        ("ul", "side-menu-ul side-menu-ul--padded index-browse-subjects hidden"),
    )
    subject_search_urls = [
        (
            "https://www.mdpi.com/search?sort=pubdate&page_no=",
            "&subjects=%s&page_count=50" % sub_url.split("/")[-1],
            sub_url.split("/")[-1],
        )
        for sub_url in sub_urls
    ]
    all_docs = []

    # Iterate all subjects
    for subject_search_url in subject_search_urls:
        GENRE_LEN = 0
        subject = subject_search_url[2]
        subject_url = []
        # if subject == "physics-astronomy":
        for n in range(1, 51):  # The number of search pages
            search_page = subject_search_url[0] + str(n) + subject_search_url[1]
            searched_urls = get_url(
                search_page, 'href="(.*?)"', ("a", "title-link")
            )  # Get 50 articles in one search page for a subject
            subject_url += searched_urls
        sys.stderr.write(f"o Finished searching articles for {subject}\n")

        # Iterate all articles within one subject
        for art_url in subject_url:
            if art_url not in prev_articles:
                prev_articles.append(art_url)
                possible_texts = get_fulltext(subject, art_url)  # Get texts

                if (
                    possible_texts is not None
                    and len(re.findall("[0-9]", possible_texts.text)) <= 40
                ):  # kick out formula-heavy articles
                    # print(len(re.findall("[0-9]", possible_texts.text)))	# the number of numericals in the article
                    all_docs.append(possible_texts)
                    GENRE_LEN += possible_texts.text.count(" ")
                    count += 1

                    possible_texts.serialize()  # write file to out/academic
                    write_file(RAW_PATH, possible_texts)  # write file to data/academic

                    if GENRE_LEN >= GENRE_LIMIT:
                        break
        TEXT_LEN += GENRE_LEN
        sys.stderr.write(f"o Finished the subject {subject} with {GENRE_LEN} tokens.\n")
    return TEXT_LEN


if __name__ == "__main__":
    url = "https://www.mdpi.com"
    sys.stderr.write("o Scraping articles\n")
    num_toks = get_texts(url)
    sys.stderr.write(f"o Done! Scraped {num_toks} tokens.\n")
