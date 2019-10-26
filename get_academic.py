import urllib.request, bz2, io, os, sys, re, shutil, random
#from dateparser.search import search_dates  # This one is worse than datefinder
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


RAW_PATH = "data" + os.sep + "academic"
OUT_PATH = "out" + os.sep + "academic"
subcorp_limit = 55000
passage_limit = 820

seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep


class Document:

    def __init__(self):
        self.title = ""
        self.subject = ""
        self.author = ""
        self.date = ""
        self.publisher = ""
        self.text = ""
        self.url = ""
        self.raw_text = ""
        self.docnum = 0

    def serialize(self, path, if_raw=True):
        if if_raw:
            text = self.raw_text
            title = "_".join(self.title.split()[:6])
            title = re.sub("[/\-]", "_", title)
            title = re.sub("\?|\*|\.|\#", "", title)
            PATH = path + os.sep + f"{title}.txt"
        else:
            heading = f"<text id=autogum_academic_doc{self.docnum}> title={self.title} subject={self.subject} " \
                      f"author={self.author} date={self.date} publisher={self.publisher} sourceURL={self.url}>\n"
            text = heading + self.text + "\n</text>"
            PATH = path + os.sep + f"autogum_academic_doc{self.docnum}.xml"
        with open(PATH, "w", encoding="utf8") as f:
            f.write(text)


def write_file(path, doc, if_raw=True):
    if if_raw:
        text = doc.raw_text
        title = doc.title.split()[:6]
        PATH = path + os.sep + "%s.txt" % " ".join(title)
    else:
        heading = f"<text id=autogum_academic_doc{doc.docnum}> title={doc.title} subject={doc.subject} " \
                  f"author={doc.author} date={doc.date} publisher={doc.publisher} sourceURL={doc.url}>\n"
        text = heading + doc.text + "\n</text>"
        PATH = path + os.sep + f"autogum_academic_doc{doc.docnum}.xml"
    with open(PATH, "w", encoding="utf8") as f:
        f.write(text)


def soup(url):
    html = requests.get(url).text
    parseed = BeautifulSoup(html)
    return parseed


def clean_text(text):
    # split the text by sections, randomly select one section
    if text.startswith("<div class=\"html-p\""):
        text = text
    else:
        index = random.randint(0, len(re.findall("data-nested=\"1\"", text))-1)
        sep = re.split("<section id=\".{1,30}\" type=\".{1,30}\"?><h2 data-nested=\"1\">", text)[1:]
        if index >= len(sep):
            index = 0
        text = re.split("<section id=\".{1,30}\" type=\".{1,30}\"?><h2 data-nested=\"1\">", text)[1:][index]

    text = re.sub("<a href=(\".*?\").*?>(.*?)</a>", "<ref target=\g<1>>\g<2></ref>", text)  # href
    text = re.sub("<a class=\"html-fig\".*?>(.*?)</a>", "<figure>\g<1></figure>", text) # figure
    text = re.sub(" \[<a.*?\]", "", text)  # reference
    text = re.sub("</?a.*?>", "", text)
    text = re.sub("</?su[b|p]>", "", text)  # special characters
    text = re.sub("<math[\w\W]*?/math>", "", text)  # math

    text = re.sub("(</?h).*?>", "\g<1>ead>", text)    # head
    text = re.sub("(</head>)(<head>)", "\g<1>\n\g<2>", text)    # head
    text = re.sub("</?section.*?>", "", text)    # section

    text = re.sub("<div class=\"html-p\"><ul class=\".*?\">", "<list>\n", text)   # list head
    text = re.sub("</ul></div>", "</list>\n", text)   # list tail
    text = re.sub("<li><div class=\"html-p\">", "<item>", text)   # item head
    text = re.sub("</div></li>", "</item>\n", text)   # item tail

    text = re.sub("</div><div class=\"html-p\">", " </p> \n<p>", text)  # paragraph
    text = re.sub("</div>(<head>)", " </p> \n\g<1>", text)  # paragraph
    text = re.sub("<div class=\"html-p\">", "\n<p>\n", text)    # paragraph

    text = re.sub("<span class=\"html-italic\">(.*?)</span>", "<hi rend=\"italic\">\g<1></hi>", text)   # italic
    text = re.sub("</?(di|a|m|span|label).*?>", "", text)    # clean-up
    text = "<head>" + text
    text += " </p>"

    passage = ""
    for p in text.split("</p>")[:-1]:
        if len(passage.split() + p.split()) <= passage_limit:
            passage = passage + p + "</p>\n"
        else:
            break
    return passage


def get_texts(url):
    count = 0

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

    '''
    def get_subject_url(url):
        parsed_html = soup(url)
        subjects_html = parsed_html.find_all('ul', 'side-menu-ul side-menu-ul--padded index-browse-subjects hidden')
        subjects_href = re.findall("href=\"(/subject.*)\"", str(subjects_html))
        subjects_url = [url+subject for subject in subjects_href]
        return subjects_url
        

    # Get first xx articles within one subject
    def get_article_url(url, art_url):
        parsed_html = soup(art_url)
        articles_html = parsed_html.find_all('a', 'title-link')
        articles_href = re.findall("href=\"(.*?)\"", str(articles_html))
        articles_url = [url+article for article in articles_href]
        return articles_url
    '''

    # Get full text of an article if it has html format to view
    def get_fulltext(subject, article_url):
        parsed_html = soup(article_url)
        article_html = parsed_html.find(id="html_link")
        if article_html:
            current_doc = Document()
            parsed_html = soup(article_url+"/htm")

            title = re.search("<div id=\"html-article-title\">(.*)</div>", str(parsed_html.find_all(id="html-article-title")))
            if title is not None:
                title = title.group(1)
            else:
                title = ""
            current_doc.title = BeautifulSoup(title, "lxml").text
            current_doc.subject = subject
            current_doc.author = ", ".join(re.findall("<meta content=\"(.*?)\" name=\".*?creator\"/>", str(parsed_html)))
            current_doc.date = re.search("<meta content=\"(.*?)\" name=\".*?date\"/>", str(parsed_html)).group(1)
            current_doc.publisher = re.search("<meta content=\"(.*?)\" name=\".*?publisher\"/>", str(parsed_html)).group(1)

            raw_text = re.search("<div class=\"html-body\">\n([\w\W]*)</div>", str(parsed_html.findAll("div",{"class":"html-body"}))).group(1)
            current_doc.raw_text = BeautifulSoup(raw_text, "lxml").text
            current_doc.text = clean_text(raw_text)

            current_doc.docnum = count
            current_doc.url = article_url + "/htm"
            return current_doc

        else:
            return None


    sub_urls = get_url(url, "href=\"(/subject.*)\"", ('ul', 'side-menu-ul side-menu-ul--padded index-browse-subjects hidden'))
    subject_search_urls = [("https://www.mdpi.com/search?sort=pubdate&page_no=", "&subjects=%s&page_count=50" % sub_url.split("/")[-1], sub_url.split("/")[-1]) for sub_url in sub_urls]
    all_docs = []

    # Iterate all subjects
    for subject_search_url in subject_search_urls:
        subject_corpora_len = 0
        subject = subject_search_url[2]
        subject_url = []
        # if subject == "computer-math":
        for n in range(1, 51):   # The number of search pages
            search_page = subject_search_url[0] + str(n) + subject_search_url[1]
            searched_urls = get_url(search_page, "href=\"(.*?)\"", ('a', 'title-link'))    # Get 50 articles in one search page for a subject
            subject_url += (searched_urls)
        sys.stderr.write(f"o Finished searching articles for {subject}\n")

        # Iterate all articles within one subject
        for art_url in subject_url:
            possible_texts = get_fulltext(subject, art_url)     # Get texts
            if possible_texts is not None:
                all_docs.append(possible_texts)
                count += 1

                possible_texts.serialize(RAW_PATH, if_raw=True)  # write file to data/academic
                possible_texts.serialize(OUT_PATH, if_raw=False) # write file to out/academic

                subject_corpora_len += len(possible_texts.text.split())
                if subject_corpora_len >= subcorp_limit:
                    break
        sys.stderr.write(f"o Finished the subject {subject}\n")


if __name__ == "__main__":
    url = "https://www.mdpi.com"
    sys.stderr.write("o Scraping articles\n")
    get_texts(url)
    sys.stderr.write("o Done!\n")
