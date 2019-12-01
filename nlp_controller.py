# -*- coding: utf-8 -*-

"""
nlp_controller.py
A general purpose interface to add annotations to data coming from all genres
"""

import stanfordnlp
import os, io, sys, re, platform
from xml.dom import minidom
from argparse import ArgumentParser
from lib.whitespace_tokenize import tokenize as tt_tokenize
from lib.utils import exec_via_temp, get_col
from glob import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
LIB_DIR = SCRIPT_DIR + "lib" + os.sep
BIN_DIR = SCRIPT_DIR + "bin" + os.sep
TT_PATH = BIN_DIR + "treetagger" + os.sep + "bin" + os.sep

# Setup StanfordNLP
# Uncomment to download models
# stanfordnlp.download('en')       # This downloads the English models for the neural pipeline
# stanfordnlp.download('en_gum')   # This adds gum models

STANFORD_NLP_CONFIG = {
    "lang": "en",
    "treebank": "en_gum",
    "processors": "tokenize,pos,lemma,depparse",
    "tokenize_pretokenized": True,
    "pos_batch_size": 500,  # 1000,
    "depparse_batch_size": 500
    # We could possibly mix and match models here, but it is probably a bad idea
    #  'pos_model_path': 'en_ewt_models/en_ewt_tagger.pt',
    #  'pos_pretrain_path': 'en_ewt_models/en_ewt.pretrain.pt',
    #  'lemma_model_path': 'en_ewt_lemmatizer/en_ewt_lemmatizer.pt',
    #  'depparse_model_path': 'en_ewt_lemmatizer/en_ewt_parser.pt',
    #  'depparse_pretrain_path': 'en_ewt_lemmatizer/en_ewt.pretrain.pt'
}


def postprocess_text(TTSGML):
    # Likely verbal VVN, probably not an amod
    VVN = "been|called|made|found|seen|done|based|taken|born|considered|got|located|said|told|started|shown|become|put|gone|created|had|asked"
    ART = "the|this|that|those|a|an"

    # Phone numbers
    phone_exp = re.findall(
        r"((?:☏|(?:fax|phone)\n:)\n(?:\+?[0-9]+\n|-\n)+)", TTSGML, flags=re.UNICODE
    )
    for phone in phone_exp:
        fused = (
            phone.replace("\n", "")
            .replace("☏", "☏\n")
            .replace("fax:", "fax\n:\n")
            .replace("phone:", "phone\n:\n")
            + "\n"
        )
        TTSGML = TTSGML.replace(phone, fused)

    # Currency
    TTSGML = re.sub(r"([¥€\$])([0-9,.]+)\n", r"\1\n\2\n", TTSGML)

    # Ranges
    TTSGML = re.sub(
        r"(¥|\$|€)\n?([0-9.,]+)-([0-9.,]+\n)", r"\1\n\2\n-\n\3", TTSGML
    )  # Currency
    TTSGML = re.sub(
        r"([12]?[0-9]:[0-5][0-9])(-)([12]?[0-9]:[0-5][0-9])\n", r"\1\n\2\n\3\n", TTSGML
    )  # Time
    TTSGML = re.sub(
        r"((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)-((?:sun|mon|tues?|wed|thu(?:rs)|fri|sat(?:ur)?)(?:day)?)\n",
        r"\1\n-\n\2\n",
        TTSGML,
        flags=re.IGNORECASE,
    )  # Days
    TTSGML = re.sub(
        r"(Su|M|Tu|W|Th|Fr?|Sa)-(Su|M|Tu|W|Th|Fr?|Sa)\n", r"\1\n-\n\2\n", TTSGML
    )  # Short days

    # Measurement symbols
    TTSGML = re.sub(r"\n(k?m)²\n", r"\n\1\n²\n", TTSGML)  # Squared
    TTSGML = re.sub(r"([0-9])°\n", r"\1\n°", TTSGML)  # Degree symbol

    # Latin abbreviations
    TTSGML = TTSGML.replace(" i. e. ", " i.e. ").replace(" e. g. ", " e.g. ")

    # Trailing periods in section headings like "1. Introduction", usually following an XML tag
    TTSGML = re.sub(r"(>\n[0-9]+)\n(\.\n)", r"\1\2", TTSGML)

    # en dash spelled --
    TTSGML = re.sub(r"([^\n])--", r"\1\n--", TTSGML)
    TTSGML = re.sub(r"--([^\n]+)\n", r"--\n\1\n", TTSGML)

    # Find missing contraction spellings
    TTSGML = re.sub(r"\n([Ii]t)(s\nnot\n)", r"\n\1\n\2", TTSGML)
    TTSGML = TTSGML.replace("\nIve\n", "\nI\nve\n")
    TTSGML = re.sub(
        r"\n(did|do|was|were|would|should|had|must)nt\n", r"\n\1\nnt\n", TTSGML
    )

    # Fix grammar-dependant tokenizations
    TTSGML = re.sub(r"(\n[Ii]t)(s\n(?:" + VVN + ART + r")\n)", r"\1\n\2", TTSGML)

    # Fix apostrophes
    TTSGML = re.sub(r">\n'\ns\n", r">\n's\n", TTSGML)

    # Fix parentheses in tokens like parent(s)
    TTSGML = re.sub(r"\n(\w+)\(s\n\)\n", r"\n\1(s)\n", TTSGML)

    fixed = TTSGML
    return fixed


def escape_treetagger(tt_sgml):
    new_lines = []
    for line in tt_sgml.split("\n"):
        # probably an element
        if line.startswith("<") and line.endswith(">"):
            new_lines.append(line)
        else:
            new_lines.append(
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )
    return "\n".join(new_lines)


def tokenize(xml_data):
    """Tokenize input XML or plain text into TT SGML format.

    :param xml_data: input string of a single document
    :return: TTSGML with exactly one token or opening tag or closing tag per line

    example input:

        <text id="autogum_voyage_doc3" title="Aakirkeby">
        <head>Aakirkeby</head>

        <p><hi rend="bold">Aakirkeby</hi> is a city on <ref target="Bornholm">Bornholm</ref>,

    example output:

        <text id="autogum_voyage_doc3" title="Aakirkeby">
        <head>
        Aakirkeby
        </head>
        <p>
        <hi rend="bold">
        Aakirkeby
        </hi>
        is
        a
        city
        ...
    """
    # Separate en/em dashes
    xml_data = xml_data.replace("–", " – ").replace("—", " — ")

    abbreviations = LIB_DIR + "english-abbreviations"
    tokenized = tt_tokenize(xml_data, abbr=abbreviations)

    def postprocess_text_nodes(node):
        if hasattr(node, "childNodes") and list(node.childNodes):
            for child in node.childNodes:
                postprocess_text_nodes(child)
        elif node.nodeType == minidom.Node.TEXT_NODE:
            node.data = postprocess_text(node.data)

    # TreeTagger doesn't escape XML chars. We need to fix it before we parse.
    tokenized = escape_treetagger(tokenized)

    xml = minidom.parseString(tokenized)
    postprocess_text_nodes(xml)
    tokenized = xml.toxml()

    return tokenized


def tt_tag_command():
    if not os.path.exists(TT_PATH + "tree-tagger"):
        sys.stderr.write(
            "TreeTagger binary not available at " + TT_PATH + "tree-tagger\n"
        )
        sys.stderr.write(
            "Download TreeTagger binary from https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/\n"
        )
        sys.exit(1)
    if not os.path.exists(TT_PATH + "english.par"):
        sys.stderr.write(
            "TreeTagger English parameter file not found at "
            + TT_PATH
            + "english.par\n"
        )
        sys.stderr.write(
            "Download English PTB parameter file from https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz\n"
        )
        sys.exit(1)
    tag = [
        TT_PATH + "tree-tagger",
        TT_PATH + "english.par",
        "-lemma",
        "-no-unknown",
        "-sgml",
        "tempfilename",
    ]
    return tag


def marmot_tag_command():
    jar_sep = ";" if platform.system() == "Windows" else ":"
    tag = [
        "java",
        "-Dfile.encoding=UTF-8",
        "-Xmx2g",
        "-cp",
        "marmot.jar" + jar_sep + "trove.jar",
        "marmot.morph.cmd.Annotator",
        "-model-file",
        "eng.marmot",
        "-lemmatizer-file",
        "eng.lemming",
        "-test-file",
        "form-index=0,tempfilename",
        "-pred-file",
        "tempfilename2",
    ]
    return tag


def pos_tag(tokenized_document):
    data = tokenized_document.strip().split("\n")
    orig_data_split = data
    data = [
        word.strip().replace("’", "'") for word in data
    ]  # Use plain quotes for easier tagging
    data = "\n".join(data)

    # Choose tagger here
    tagger = "TT"
    outfile = False

    if tagger == "TT":
        tag = tt_tag_command()
    elif tagger == "marmot":
        outfile = True
        tag = marmot_tag_command()
    else:
        raise Exception("Unknown tagger: " + tagger)

    tagged = exec_via_temp(data, tag, outfile=outfile)
    tagged = tagged.strip().replace("\r", "").split("\n")
    if tagger == "marmot":
        tags = get_col(tagged, 5)
        lemmas = get_col(tagged, 3)
    else:
        tags = get_col(tagged, 0)
        lemmas = get_col(tagged, 1)

    assert len(tags) == len(lemmas)
    assert len(tags) > 0

    def postprocess(node):
        nonlocal counter
        if hasattr(node, "childNodes") and list(node.childNodes):
            for child in node.childNodes:
                postprocess(child)
        elif node.nodeType == minidom.Node.TEXT_NODE:
            lines = node.data
            outlines = []
            for line in lines.split("\n"):
                if line.strip() == "":
                    outlines.append(line)
                    continue
                else:
                    pos, lemma = tags[counter], lemmas[counter]
                    lemma = lemma.replace('"', "''")
                    if line.strip() == "“":
                        pos = "``"
                    elif line.strip() == "”":
                        pos = "''"
                    elif line.strip() == "[":
                        pos = "("
                    elif line.strip() == "]":
                        pos = ")"
                    outlines.append("\t".join([line, pos, lemma]))
                    counter += 1
            node.data = "\n".join(outlines)

    data = orig_data_split
    xml = minidom.parseString("\n".join(data))
    counter = 0
    postprocess(xml)
    tagged = xml.toxml()

    return tagged


def dep_parse(tokenized, snlp, torch):
    # StanfordNLP expects a list of sentences, each a list of token strings, in order to run in pre-tokenized mode
    sent_list = [s.strip().split() for s in tokenized.strip().split("\n")]
    torch.cuda.empty_cache()

    doc = snlp(sent_list)

    return doc.conll_file.conll_as_string()


def process_genre(
    genre,
    opts,
    XML_OUTPUT_DIR=SCRIPT_DIR + "nlped" + os.sep + "xml" + os.sep,
    DEPENDENCY_OUTPUT_DIR=SCRIPT_DIR + "nlped" + os.sep + "dep" + os.sep,
):
    # set up stanford nlp if we need it for dep parsing
    if not opts.no_parse:
        # only import torch if we need it
        import torch

        torch.cuda.init()

        # default english pipeline
        snlp = stanfordnlp.Pipeline(**STANFORD_NLP_CONFIG, use_gpu=True)

    # the files we need to process
    INPUT_FILE_DIR = SCRIPT_DIR + "out" + os.sep + genre + os.sep + "autogum_*.xml"
    filepaths = sorted(glob(INPUT_FILE_DIR))

    for file_num, filepath in enumerate(filepaths):
        with open(filepath, "r", encoding="utf8") as f:
            raw_xml = f.read()
        tokenized = tokenize(raw_xml)

        tok_count = len(
            [t for t in tokenized.strip().split("\n") if not t.startswith("<")]
        )

        sys.stderr.write(
            "o Processing document: "
            + os.path.basename(filepath)
            + " ("
            + str(file_num + 1)
            + "/"
            + str(len(filepaths))
            + ")"
        )

        # Skip documents that are way too big or small
        if tok_count < 300 or tok_count > 2000:
            sys.stderr.write(" [skipped due to size]\n")
            continue
        else:
            sys.stderr.write("\n")

        # POS tag
        # If we want to tag outside StanfordNLP, a dedicated tagger can be used
        if opts.no_parse:
            tagged = pos_tag(tokenized)

        # Add sentence splits - note this currently produces mal-nested SGML
        if opts.no_sent:
            split_indices = [1] + [0] * (len(tokenized) - 1)
        else:
            from lib.gumdrop.EnsembleSentencer import EnsembleSentencer

            best_sentencer_ever = EnsembleSentencer(
                lang="eng", model="eng.rst.gum", genre_pat="_([^_]+)_"
            )
            split_indices = best_sentencer_ever.predict(
                tokenized, as_text=True, plain=True, genre=genre
            )

        counter = 0
        splitted = []
        opened_sent = False
        para = True
        for line in tokenized.strip().split("\n"):
            if not (line.startswith("<") and line.endswith(">")):
                # Token
                if split_indices[counter] == 1 or para:
                    if opened_sent:
                        splitted.append("</s>")
                        opened_sent = False
                    splitted.append("<s>")
                    opened_sent = True
                    para = False
                counter += 1
            elif (
                "<p>" in line or "<head>" in line or "<caption>" in line
            ):  # New block, force sentence split
                para = True
            splitted.append(line)
        splitted = "\n".join(splitted)
        if opened_sent:
            if splitted.endswith("</text>"):
                splitted = splitted.replace("</text>", "</s>\n</text>")
            else:
                splitted += "\n</s>"

        if not opts.no_parse:
            # Parse
            no_xml = splitted.replace("</s>\n<s>", "---SENT---")
            no_xml = re.sub(r"<[^<>]+>\n?", "", no_xml)

            sents = no_xml.strip().replace("\n", " ").replace("---SENT--- ", "\n")
            parsed = dep_parse(sents, snlp, torch)
        else:
            parsed = tagged

        doc = os.path.basename(filepath)

        # Insert tags into XML
        pos_lines = []
        lemma_lines = []
        for line in parsed.split("\n"):
            if "\t" in line:
                fields = line.split("\t")
                if opts.no_parse:
                    lemma, xpos = fields[2], fields[1]
                else:
                    lemma, xpos = fields[2], fields[4]
                pos_lines.append(xpos)
                lemma_lines.append(lemma)
        tagged = []
        counter = 0
        for line in splitted.split("\n"):
            if line.startswith("<") and line.endswith(">"):
                tagged.append(line)
            else:
                line = line + "\t" + pos_lines[counter] + "\t" + lemma_lines[counter]
                tagged.append(line)
                counter += 1
        tagged = "\n".join(tagged)

        # Write output files
        with io.open(XML_OUTPUT_DIR + doc, "w", encoding="utf8", newline="\n") as f:
            f.write(tagged)

        if not opts.no_parse:
            with io.open(
                DEPENDENCY_OUTPUT_DIR + doc.replace(".xml", ".conllu"),
                "w",
                encoding="utf8",
                newline="\n",
            ) as f:
                f.write(parsed)

        # TODO: entities + coref

        # TODO: RST


def main():
    p = ArgumentParser()
    p.add_argument(
        "--no_parse", action="store_true", help="do not perform dependency parsing"
    )
    p.add_argument(
        "--no_sent", action="store_true", help="do not perform sentence splitting"
    )
    p.add_argument(
        "-g",
        "--genre",
        action="store",
        choices=[
            "news",
            "interview",
            "bio",
            "academic",
            "voyage",
            "whow",
            "reddit",
            "fiction",
        ],
        help="only process this genre",
    )
    opts = p.parse_args()
    if opts.no_sent:
        opts.no_parse = True

    if opts.genre is not None:
        process_genre(opts.genre, opts)
    else:
        for genre in [
            "news",
            "interview",
            "bio",
            "academic",
            "voyage",
            "whow",
            "reddit",
            "fiction",
        ]:
            process_genre(genre, opts)


if __name__ == "__main__":
    main()
