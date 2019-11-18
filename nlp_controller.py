#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
nlp_controller.py
A general purpose interface to add annotations to data coming from all genres
"""

import stanfordnlp
import os, io, sys, re, platform
from argparse import ArgumentParser
from lib.whitespace_tokenize import tokenize as tt_tokenize
from lib.utils import exec_via_temp, get_col
from lib.gumdrop.EnsembleSentencer import EnsembleSentencer
from glob import glob

import torch

torch.cuda.init()

PY3 = sys.version_info[0] == 3

# Paths
script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
lib_dir = script_dir + "lib" + os.sep
bin_dir = script_dir + "bin" + os.sep
xml_out = script_dir + "nlped" + os.sep + "xml" + os.sep
dep_out = script_dir + "nlped" + os.sep + "dep" + os.sep

# Setup StanfordNLP
# Uncomment to download models
# stanfordnlp.download('en')       # This downloads the English models for the neural pipeline
# stanfordnlp.download('en_gum')   # This adds gum models

config = {
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


class AutoGumNLP:
    def __init__(self):
        pass

    def pos_tag(self, tokenized_document):
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
            tt_path = bin_dir + "treetagger" + os.sep + "bin" + os.sep
            tag = [
                tt_path + "tree-tagger",
                tt_path + "english.par",
                "-lemma",
                "-no-unknown",
                "-sgml",
                "tempfilename",
            ]
        elif tagger == "marmot":
            outfile = True
            if platform.system() == "Windows":
                tag = [
                    "java",
                    "-Dfile.encoding=UTF-8",
                    "-Xmx2g",
                    "-cp",
                    "marmot.jar;trove.jar",
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
            else:
                tag = [
                    "java",
                    "-Dfile.encoding=UTF-8",
                    "-Xmx2g",
                    "-cp",
                    "marmot.jar:trove.jar",
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

        tagged = exec_via_temp(data, tag, outfile=outfile)
        tagged = tagged.strip().replace("\r", "").split("\n")
        if tagger == "marmot":
            tags = get_col(tagged, 5)
            lemmas = get_col(tagged, 3)
        else:
            tags = get_col(tagged, 0)
            lemmas = get_col(tagged, 1)

        data = orig_data_split

        outlines = []
        counter = 0
        for i, line in enumerate(data):
            if not line.startswith("<") and not line.endswith(">"):
                pos, lemma = tags[counter], lemmas[counter]
                lemma = lemma.replace('"', "''")
                if data[i].strip() == "“":
                    pos = "``"
                elif data[i].strip() == "”":
                    pos = "''"
                elif data[i].strip() == "[":
                    pos = "("
                elif data[i].strip() == "]":
                    pos = ")"
                outline = "\t".join([line, pos, lemma])
                outlines.append(outline)
                counter += 1
            else:
                outlines.append(line)
        tagged = "\n".join(outlines)

        return tagged

    def dep_parse(self, tokenized):

        # StanfordNLP expects a list of sentences, each a list of token strings, in order to run in pre-tokenized mode
        sent_list = [s.strip().split() for s in tokenized.strip().split("\n")]
        torch.cuda.empty_cache()

        doc = self.stan(sent_list)

        return doc.conll_file.conll_as_string()

    def tokenize(self, xml_data):
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

        # Likely verbal VVN, probably not an amod
        VVN = "been|called|made|found|seen|done|based|taken|born|considered|got|located|said|told|started|shown|become|put|gone|created|had|asked"
        ART = "the|this|that|those|a|an"

        def postprocess_tok(TTSGML):
            # Phone numbers
            phone_exp = re.findall(
                r"((?:☏|(?:fax|phone)\n:)\n(?:\+?[0-9]+\n|-\n)+)",
                TTSGML,
                flags=re.UNICODE,
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
                r"([12]?[0-9]:[0-5][0-9])(-)([12]?[0-9]:[0-5][0-9])\n",
                r"\1\n\2\n\3\n",
                TTSGML,
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

            # en dash spelled --
            TTSGML = re.sub(r"([^\n])--", r"\1\n--", TTSGML)
            TTSGML = re.sub(r"--([^\n]+)\n", r"--\n\1\n", TTSGML)

            # Find missing contraction spellings
            TTSGML = re.sub(r"\n([Ii]t)(s\nnot\n)", r"\n\1\n\2", TTSGML)
            TTSGML = TTSGML.replace("\nIve\n", "\nI\nve\n")

            # Fix grammar-dependant tokenizations
            TTSGML = re.sub(
                r"(\n[Ii]t)(s\n(?:" + VVN + ART + r")\n)", r"\1\n\2", TTSGML
            )

            # Fix apostrophes
            TTSGML = re.sub(r">\n'\ns\n", r">\n's\n", TTSGML)

            # Fix parentheses in tokens like parent(s)
            TTSGML = re.sub(r"\n(\w+)\(s\n\)\n", r"\n\1(s)\n", TTSGML)

            fixed = TTSGML
            return fixed

        abbreviations = lib_dir + "english-abbreviations"

        # Separate n/m dashes
        xml_data = xml_data.replace("–", " – ").replace("—", " — ")

        tokenized = tt_tokenize(xml_data, abbr=abbreviations)
        tokenized = postprocess_tok(tokenized)

        return tokenized


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

    nlp = AutoGumNLP()

    if opts.no_sent:
        opts.no_parse = True
    if not opts.no_parse:
        stan = stanfordnlp.Pipeline(
            **config, use_gpu=True
        )  # This sets up a default neural pipeline in English
        nlp.stan = stan

    genre = "voyage"
    test = script_dir + "out" + os.sep + genre + os.sep + "autogum_*.xml"

    files = glob(test)

    for file_num, file_ in enumerate(files):
        raw_xml = io.open(file_, encoding="utf8").read()
        tokenized = nlp.tokenize(raw_xml)

        tok_count = len(
            [t for t in tokenized.strip().split("\n") if not t.startswith("<")]
        )

        sys.stderr.write(
            "o Processing document: "
            + os.path.basename(file_)
            + " ("
            + str(file_num + 1)
            + "/"
            + str(len(files))
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
            tagged = nlp.pos_tag(tokenized)

        # Add sentence splits - note this currently produces mal-nested SGML
        if opts.no_sent:
            split_indices = [1] + [0] * (len(tokenized) - 1)
        else:
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
            parsed = nlp.dep_parse(sents)
        else:
            parsed = tagged

        doc = os.path.basename(file_)

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
        with io.open(xml_out + doc, "w", encoding="utf8", newline="\n") as f:
            f.write(tagged)

        if not opts.no_parse:
            with io.open(
                dep_out + doc.replace(".xml", ".conllu"),
                "w",
                encoding="utf8",
                newline="\n",
            ) as f:
                f.write(parsed)

        # TODO: entities + coref

        # TODO: RST


if __name__ == "__main__":
    main()
