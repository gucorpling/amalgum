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

rst_dir = "C:\\uni\\corpora\\gum\\github\\_build\\target\\rst\\rstweb\\"
conll_dir = "C:\\uni\\corpora\\gum\\github\\_build\\target\\dep\\not-to-release\\"

files = glob(rst_dir + "*.rs3")

data = defaultdict(list)

for file_ in files:
    doc = os.path.basename(file_).replace(".rs3", "")
    conll_file = conll_dir + doc + ".conllu"

    partition = "train"
    if doc in test:
        partition = "test"
    elif doc in dev:
        partition = "dev"

    lines = io.open(file_, encoding="utf8").read().strip().split("\n")

    seg = "B-SEG"
    rst_segs = []
    rst_tokens = []
    for line in lines:
        if "<segment" in line:
            toks = re.search(r'<segment[^<>\n]*>([^<>]+)</segment>',line).group(1).strip().split()
            for i ,tok in enumerate(toks):
                if i == 0:
                    seg = "B-SEG"
                else:
                    seg = "O"
                rst_segs.append(seg)
                rst_tokens.append(tok)

    conll = io.open(conll_file, encoding="utf8").read().strip()
    conll_toks = []
    sents = conll.split("\n\n")
    sent_tokens = defaultdict(list)
    for i, sent in enumerate(sents):
        toks = []
        for line in sent.split("\n"):
            if "\t" in line:
                fields = line.split("\t")
                if "-" in fields[0]:
                    continue
                toks.append(fields[1])
                conll_toks.append(fields[1])
        sent_tokens[i] = toks

    pre_context = []
    post_context = []
    #output = ["-DOCSTART- X",""]
    output = []
    counter = 0
    for i, sent in enumerate(sents):
        prev_s = sent_tokens[i-1] if i > 0 else sent_tokens[len(sents)-1]  # Use last sent as prev if this is sent 1
        pre_context = prev_s[-6:] if len(prev_s) > 5 else prev_s[:]
        pre_context.append("<pre>")
        next_s = sent_tokens[i+1] if i < len(sents)-1 else sent_tokens[0]  # Use first sent as next if this is last sent
        post_context = next_s[:6] if len(next_s) > 5 else next_s[:]
        post_context = ["<post>"] + post_context

        for tok in pre_context:
            output.append(tok+"\t" + "O")
        for tok in sent_tokens[i]:
            output.append(tok+"\t" + rst_segs[counter])
            counter += 1
        for tok in post_context:
            output.append(tok+"\t" + "O")
        output.append("")

    data[partition] += output

for partition in data:
    lines = data[partition]
    with io.open("edu_" + partition + ".txt", "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
