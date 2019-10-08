import jieba
from itertools import chain
from collections import Counter
from nltk import word_tokenize
from argparse import ArgumentParser


def read_file(filename):

    lines = []
    with open(filename, 'r', encoding="utf-8") as f:
        file = f.readlines()
        for stuff in file:
            if stuff != "\n":
                if not stuff.startswith("</doc>"):
                    if not stuff.startswith("<doc id="):
                        lines.append(stuff.splitlines())

    texts = list(chain.from_iterable(lines))

    return texts


def tokenize_text(file, lang):

    tokens = []

    for lines in file:
        if lang == "zho":
            token = ' '.join(jieba.cut_for_search(lines))
            tokens.append(token.split(' '))
        else:
            tokens.append(word_tokenize(lines, language=lang))

    first_token_counts = Counter()
    for element in tokens:
        if len(element) != 0:
            first_token_counts[element[0]] += 1

    token_counts = Counter()
    for line in tokens:
        for tok in line:
            token_counts[tok] += 1

    first_token = []
    freq_thresh = 5
    ratio_thresh = 0.8

    for item in first_token_counts.keys():
        if first_token_counts[item] >= freq_thresh and (first_token_counts[item] / token_counts[item]) >= ratio_thresh:
            first_token.append(item)

    return first_token


eng_wiki_token = tokenize_text(read_file("eng_wiki.xml"), "english")

with open("eng.txt", 'w', encoding='utf-8') as d:
    for item in eng_wiki_token:
        doc = d.writelines(item + '\t')