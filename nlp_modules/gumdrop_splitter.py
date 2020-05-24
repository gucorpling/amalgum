import os
import re

from glob import glob
from nlp_modules.base import NLPModule, PipelineDep
import conllu
from collections import OrderedDict, defaultdict
from lib.reorder_sgml import reorder

TAGS = ["sp","table","row","cell","head","p","figure","caption","list","item","quote","s","q","hi","sic","ref","date","incident","w"]
BLOCK_TAGS = ["sp","head","p","figure","caption","list","item"]
OPEN_SGML_ELT = re.compile(r'^<([^/ ]+)( .*)?>$')
CLOSE_SGML_ELT = re.compile(r'^</([^/]+)>$')


def maximal_nontoken_span_end(sgml_list, i):
    """Return j such that sgml_list[i:j] does not contain tokens
    and no element that is begun in the MNS is closed in it."""
    opened = []
    j = i
    while j < len(sgml_list):
        line = sgml_list[j]
        open_match = re.match(OPEN_SGML_ELT, line)
        close_match = re.match(CLOSE_SGML_ELT, line)
        if not (open_match or close_match):
            break
        if open_match:
            opened.append(open_match.groups()[0])
        if close_match and close_match.groups()[0] in opened:
            break
        j += 1
    return j


def fix_malformed_sentences(sgml_list):
    """
    Fixing malformed SGML seems to boil down to two cases:

    (1) The sentence is interrupted by the close of a tag that opened before it. In this case,
        update the s boundaries so that we close and begin sentences at the close tag:

                             <a>
                <a>          ...
                ...          <s>
                <s>          ...
                ...    ==>   </s>
                </a>         </a>
                ...          <s>
                </s>         ...
                             </s>

    (2) Some tag opened inside of the sentence and has remained unclosed at the time of sentence closure.
        In this case, we choose not to believe the sentence split, and merge the two sentences:

                <s>
                ...          <s>
                <a>          ...
                ...          <a>
                </s>   ==>   ...
                <s>          ...
                ...          </a>
                </a>         ...
                ...          </s>
                </s>
    """
    tag_opened = defaultdict(list)
    i = 0
    while i < len(sgml_list):
        line = sgml_list[i].strip()
        open_match = re.search(OPEN_SGML_ELT, line)
        close_match = re.search(CLOSE_SGML_ELT, line)
        if open_match:
            tag_opened[open_match.groups()[0]].append(i)
        elif close_match:
            tagname = close_match.groups()[0]
            j = maximal_nontoken_span_end(sgml_list, i + 1)
            mns = sgml_list[i:j]

            # case 1: we've encountered a non-s closing tag. If...
            if (
                tagname != 's'                                       # the closing tag is not an s
                and len(tag_opened['s']) > 0                         # and we're in a sentence
                and len(tag_opened[tagname]) > 0 and len(tag_opened['s']) > 0  # and the sentence opened after the tag
                and tag_opened[tagname][-1] < tag_opened['s'][-1]
                and '</s>' not in mns                                # the sentence is not closed in the mns
            ):
                # end sentence here and move i back to the line we were looking at
                sgml_list.insert(i, '</s>')
                i += 1
                # open a new sentence at the end of the mns and note that we are no longer in the sentence
                sgml_list.insert(j+1, '<s>')
                tag_opened['s'].pop(-1)
                # we have successfully closed this tag
                tag_opened[tagname].pop(-1)
            # case 2: s closing tag and there's some tag that opened inside of it that isn't closed in time
            elif tagname == 's' and any(e != 's' and f'</{e}>' not in mns for e in
                                     [e for e in tag_opened.keys()
                                      if len(tag_opened[e]) > 0 and len(tag_opened['s']) > 0
                                         and tag_opened[e][-1] > tag_opened['s'][-1]]):
                # some non-s element opened within this sentence and has not been closed even in the mns
                assert '<s>' in mns
                sgml_list.pop(i)
                i -= 1
                sgml_list.pop(i + mns.index('<s>'))
            else:
                tag_opened[tagname].pop(-1)
        i += 1
    return sgml_list


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def unescape(token):
    token = token.replace("&quot;", '"')
    token = token.replace("&lt;", '<')
    token = token.replace("&gt;", '>')
    token = token.replace("&amp;", '&')
    token = token.replace("&apos;", "'")
    return token


def tokens2conllu(tokens):
    tokens = [
        OrderedDict(
            (k, v)
            for k, v in zip(
                conllu.parser.DEFAULT_FIELDS,
                [i + 1, unescape(token)]
                + ["_" for i in range(len(conllu.parser.DEFAULT_FIELDS) - 1)],
            )
        )
        for i, token in enumerate(tokens)
    ]
    tl = conllu.TokenList(tokens)
    return tl


class GumdropSplitter(NLPModule):
    requires = (PipelineDep.TOKENIZE,)
    provides = (PipelineDep.S_SPLIT, PipelineDep.S_TYPE)

    def __init__(self, config):
        self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()

    def test_dependencies(self):
        import tensorflow
        import keras
        import rfpimp
        import xgboost
        import nltk
        import pandas
        import scipy
        import torch
        import hyperopt
        import wget
        import conllu

        if tensorflow.version.VERSION[0] != "1":
            raise Exception("TensorFlow version 1.x must be used with GumdropSplitter")
        model_dir = os.path.join(self.LIB_DIR, "gumdrop", "lib", "udpipe")
        if len(glob(os.path.join(model_dir, "english-*.udpipe"))) == 0:
            raise Exception(
                "No English udpipe model found. Please download the an English udpipe model from "
                "from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131"
                f" and place it in {model_dir}/."
            )

    def split(self, context):
        xml_data = context["xml"]
        # Sometimes the tokenizer doesn't newline every elt
        xml_data = xml_data.replace('><', '>\n<')
        # Ad hoc fix for a tokenization error
        xml_data = xml_data.replace('°<', '°\n<')
        # Remove empty elements?
        #for elt in TAGS:
        #    xml_data = xml_data.replace(f"<{elt}>\n</{elt}>\n", "")

        # Search for genre in the first 2 lines (in case there's an <?xml version="1.0" ?>
        genre = re.findall(r'type="(.*?)"', "\n".join(xml_data.split("\n")[:2]))
        assert len(genre) == 1
        genre = genre[0]
        # don't feed the sentencer our pos and lemma predictions, if we have them
        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)
        split_indices = self.sentencer.predict(
            no_pos_lemma, as_text=True, plain=True, genre=genre
        )

        # for xml
        counter = 0
        splitted = []
        opened_sent = False
        para = True

        for line in xml_data.strip().split("\n"):
            if not is_sgml_tag(line):
                # Token
                if split_indices[counter] == 1 or para:
                    if opened_sent:
                        rev_counter = len(splitted) - 1
                        while is_sgml_tag(splitted[rev_counter]):
                            rev_counter -= 1
                        splitted.insert(rev_counter + 1, "</s>")
                    splitted.append("<s>")
                    opened_sent = True
                    para = False
                counter += 1
            elif (
                any(f'<{elt}>' in line for elt in BLOCK_TAGS)
                or any(f'</{elt}>' in line for elt in BLOCK_TAGS)
            ):  # New block, force sentence split
                para = True
            splitted.append(line)

        if opened_sent:
            rev_counter = len(splitted) - 1
            while is_sgml_tag(splitted[rev_counter]):
                rev_counter -= 1
            splitted.insert(rev_counter + 1, "</s>")

        lines = "\n".join(splitted)
        lines = reorder(lines)
        lines = fix_malformed_sentences(lines.split("\n"))
        lines = "\n".join(lines)
        lines = reorder(lines)

        # now, we need to construct the sentences for conllu
        conllu_sentences = []
        tokens = []
        in_sent = False
        for i, line in enumerate(lines.strip().split("\n")):
            if line == '<s>':
                in_sent = True
                if len(tokens) > 0:
                    conllu_sentences.append(tokens2conllu(tokens))
                    tokens = []
            elif line == '</s>':
                in_sent = False
            elif not is_sgml_tag(line):
                if not in_sent:
                    raise Exception(f"Encountered a token '{line}' not in a sentence at line {i}")
                else:
                    tokens.append(line)
        if len(tokens) > 0:
            conllu_sentences.append(tokens2conllu(tokens))

        return {
            "xml": lines,
            "dep": "\n".join(tl.serialize() for tl in conllu_sentences),
        }

    def run(self, input_dir, output_dir):
        from lib.gumdrop.EnsembleSentencer import EnsembleSentencer

        self.sentencer = EnsembleSentencer(
            lang="eng", model="eng.rst.gum", genre_pat="_([^_]+)_"
        )

        # Identify a function that takes data and returns output at the document level
        processing_function = self.split

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function, multithreaded=False
        )
