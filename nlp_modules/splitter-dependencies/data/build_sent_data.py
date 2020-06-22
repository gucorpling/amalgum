from glob import glob
import os, sys, re, io
from collections import defaultdict

dev = [
    "GUM_interview_peres",
    "GUM_interview_cyclone",
    "GUM_interview_gaming",
    "GUM_news_iodine",
    "GUM_news_defector",
    "GUM_news_homeopathic",
    "GUM_voyage_athens",
    "GUM_voyage_isfahan",
    "GUM_voyage_coron",
    "GUM_whow_joke",
    "GUM_whow_skittles",
    "GUM_whow_overalls",
    "GUM_fiction_beast",
    "GUM_bio_emperor",
    "GUM_academic_librarians",
    "GUM_fiction_lunre",
    "GUM_bio_byron",
    "GUM_academic_exposure",
]
test = [
    "GUM_interview_mcguire",
    "GUM_interview_libertarian",
    "GUM_interview_hill",
    "GUM_news_nasa",
    "GUM_news_expo",
    "GUM_news_sensitive",
    "GUM_voyage_oakland",
    "GUM_voyage_thailand",
    "GUM_voyage_vavau",
    "GUM_whow_mice",
    "GUM_whow_cupcakes",
    "GUM_whow_cactus",
    "GUM_fiction_falling",
    "GUM_bio_jespersen",
    "GUM_academic_discrimination",
    "GUM_academic_eegimaa",
    "GUM_bio_dvorak",
    "GUM_fiction_teeth",
]

conll_dir = "C:\\uni\\corpora\\gum\\github\\_build\\target\\dep\\not-to-release\\"

files = glob(conll_dir + "*.conllu")

data = defaultdict(list)

for file_ in files:
    doc = os.path.basename(file_).replace(".conllu", "")

    partition = "train"
    if doc in test:
        partition = "test"
    elif doc in dev:
        partition = "dev"

    lines = io.open(file_, encoding="utf8").read().strip().split("\n")

    sent = "B-SENT"
    data[partition].append("-DOCSTART- X")
    data[partition].append("")
    counter = 0
    for line in lines:
        if len(line.strip()) == 0:
            sent = "B-SENT"
            data[partition].append("")
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0]:
                continue
            word = fields[1]
            pos = fields[4]
            data[partition].append(word + " " + sent)
            sent = "O"
            counter += 1
            if counter == 21:
                data[partition].append("")
                data[partition].append("-DOCSTART- X")
                data[partition].append("")
                counter = 0

    data[partition].append("")

for partition in data:
    lines = data[partition]
    with io.open("gum6_" + partition + ".txt", "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
