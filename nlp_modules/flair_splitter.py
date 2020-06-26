from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
)
from flair.models import SequenceTagger
import flair

import os, sys, re, io

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
lib_dir = script_dir + ".." + os.sep + "lib" + os.sep
nlp_modules_dir = script_dir + ".." + os.sep + "nlp_modules" + os.sep
flair_splitter_dep_dir = nlp_modules_dir + "splitter-dependencies" + os.sep

from nlp_modules.base import NLPModule, PipelineDep
import conllu
from collections import OrderedDict, defaultdict

try:
    from lib.reorder_sgml import reorder
except ImportError:
    sys.path.append(lib_dir)

TAGS = [
    "sp",
    "table",
    "row",
    "cell",
    "head",
    "p",
    "figure",
    "caption",
    "list",
    "item",
    "quote",
    "s",
    "q",
    "hi",
    "sic",
    "ref",
    "date",
    "incident",
    "w",
]
BLOCK_TAGS = ["sp", "head", "p", "figure", "caption", "list", "item"]
OPEN_SGML_ELT = re.compile(r"^<([^/ ]+)( .*)?>$")
CLOSE_SGML_ELT = re.compile(r"^</([^/]+)>$")


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
                tagname != "s"  # the closing tag is not an s
                and len(tag_opened["s"]) > 0  # and we're in a sentence
                and len(tag_opened[tagname]) > 0
                and len(tag_opened["s"]) > 0  # and the sentence opened after the tag
                and tag_opened[tagname][-1] < tag_opened["s"][-1]
                and "</s>" not in mns  # the sentence is not closed in the mns
            ):
                # end sentence here and move i back to the line we were looking at
                sgml_list.insert(i, "</s>")
                i += 1
                # open a new sentence at the end of the mns and note that we are no longer in the sentence
                sgml_list.insert(j + 1, "<s>")
                tag_opened["s"].pop(-1)
                # we have successfully closed this tag
                tag_opened[tagname].pop(-1)
            # case 2: s closing tag and there's some tag that opened inside of it that isn't closed in time
            elif tagname == "s" and any(
                e != "s" and f"</{e}>" not in mns
                for e in [
                    e
                    for e in tag_opened.keys()
                    if len(tag_opened[e]) > 0 and len(tag_opened["s"]) > 0 and tag_opened[e][-1] > tag_opened["s"][-1]
                ]
            ):
                # some non-s element opened within this sentence and has not been closed even in the mns
                assert "<s>" in mns
                sgml_list.pop(i)
                i -= 1
                sgml_list.pop(i + mns.index("<s>"))
            else:
                tag_opened[tagname].pop(-1)
        i += 1
    return sgml_list


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def unescape(token):
    token = token.replace("&quot;", '"')
    token = token.replace("&lt;", "<")
    token = token.replace("&gt;", ">")
    token = token.replace("&amp;", "&")
    token = token.replace("&apos;", "'")
    return token


def tokens2conllu(tokens):
    tokens = [
        OrderedDict(
            (k, v)
            for k, v in zip(
                conllu.parser.DEFAULT_FIELDS,
                [i + 1, unescape(token)] + ["_" for i in range(len(conllu.parser.DEFAULT_FIELDS) - 1)],
            )
        )
        for i, token in enumerate(tokens)
    ]
    tl = conllu.TokenList(tokens)
    return tl


class FlairSplitter(NLPModule):
    requires = (PipelineDep.TOKENIZE,)
    provides = (PipelineDep.S_SPLIT, PipelineDep.S_TYPE)

    def __init__(self, config, model_path=None, span_size=20, stride_size=10):

        self.test_dependencies()
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.model = None
        self.span_size = span_size
        self.stride_size = stride_size
        self.LIB_DIR = config["LIB_DIR"]

    def load_model(self, path=None):
        model = "flair-splitter-sent.pt"
        if path is None:
            path = flair_splitter_dep_dir
        if not path.endswith(".pt"):
            if not path.endswith(os.sep):
                path += os.sep
            path += model
        if not os.path.exists(path):
            self.download_file(model, flair_splitter_dep_dir)
        self.model = SequenceTagger.load(path)

    def test_dependencies(self):
        # Check we have flair
        import flair

        # Get model if needed
        if not os.path.exists(flair_splitter_dep_dir + "flair-splitter-sent.pt"):
            self.download_file("flair-splitter-sent.pt", flair_splitter_dep_dir, subfolder="split")

    def train(self, training_dir=None):
        from flair.trainers import ModelTrainer

        if training_dir is None:
            training_dir = flair_splitter_dep_dir

        # define columns
        columns = {0: "text", 1: "ner"}

        # this is the folder in which train, test and dev files reside
        data_folder = flair_splitter_dep_dir + "data"

        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(
            data_folder,
            columns,
            train_file="train.txt",
            test_file="test.txt",
            dev_file="dev.txt",
            document_separator_token="-DOCSTART-",
        )

        print(corpus)

        tag_type = "ner"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)

        # 4. initialize embeddings
        embedding_types = [
            # WordEmbeddings('glove'),
            # comment in this line to use character embeddings
            CharacterEmbeddings(),
            # comment in these lines to use flair embeddings
            FlairEmbeddings("news-forward"),
            FlairEmbeddings("news-backward"),
            # BertEmbeddings('distilbert-base-cased')
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True,
        )

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(
            training_dir, learning_rate=0.1, mini_batch_size=16, max_epochs=20,
        )
        self.model = tagger

    def predict(self, tt_sgml, outmode="binary"):
        def is_tok(sgml_line):
            return not (sgml_line.startswith("<") and sgml_line.endswith(">"))

        def is_sent(line):
            return line in ["<s>", "</s>"] or line.startswith("<s ")

        if self.model is None:
            self.load_model()

        lines = tt_sgml.strip().split("\n")
        toks = [l for l in lines if is_tok(l)]
        toks = [re.sub(r"\t.*", "", t) for t in toks]

        # Hack tokens up into overlapping shingles
        wraparound = toks[-self.stride_size :] + toks + toks[: self.span_size]
        indices = (
            list(range(-self.stride_size, 0))
            + list(range(len(toks)))
            + list(range(len(toks), len(toks) + self.stride_size))
        )
        idx = 0
        sentences = []
        mapping = defaultdict(set)
        snum = 0
        while idx < len(toks):
            if idx + self.span_size < len(wraparound):
                span = wraparound[idx : idx + self.span_size]
            else:
                span = wraparound[idx:]
            sent = Sentence(" ".join(span))
            sentences.append(sent)
            for i in range(idx - self.stride_size, idx + self.span_size - self.stride_size):
                # start, end, snum
                if i >= 0 and i < len(toks):
                    mapping[i].add((idx - self.stride_size, idx + self.span_size - self.stride_size, snum))
            idx += self.stride_size
            snum += 1

        final_mapping = {}
        for idx in mapping:
            best = self.span_size
            for m in mapping[idx]:
                start, end, snum = m
                dist_to_end = end - idx
                dist_to_start = idx - start
                delta = abs(dist_to_end - dist_to_start)
                if delta < best:
                    best = delta
                    final_mapping[idx] = (snum, idx - start)  # Get sentence number and position in sentence

        # Predict
        preds = self.model.predict(sentences)

        labels = []
        for idx in final_mapping:
            snum, position = final_mapping[idx]
            if str(flair.__version__).startswith("0.4"):
                label = 0 if preds[snum].tokens[position].tags["ner"].value == "O" else 1
            else:
                label = 0 if preds[snum].tokens[position].labels[0].value == "O" else 1
            labels.append(label)

        if outmode == "binary":
            return labels

        # Generate edited SGML if desired
        output = []
        counter = 0
        first = True
        for line in lines:
            if is_sent(line):  # Remove existing sentence tags
                continue
            if is_tok(line):
                if labels[counter] == 1:
                    if not first:
                        output.append("</s>")
                    output.append("<s>")
                    first = False
                counter += 1
            output.append(line)
        output.append("</s>")  # Final closing </s>

        ordered = reorder("\n".join(output))

        return ordered.strip() + "\n"

    def split(self, context):
        xml_data = context["xml"]
        # Sometimes the tokenizer doesn't newline every elt
        xml_data = xml_data.replace("><", ">\n<")
        # Ad hoc fix for a tokenization error
        xml_data = xml_data.replace("°<", "°\n<")
        # Remove empty elements?
        # for elt in TAGS:
        #    xml_data = xml_data.replace(f"<{elt}>\n</{elt}>\n", "")

        # Search for genre in the first 2 lines (in case there's an <?xml version="1.0" ?>
        genre = re.findall(r'type="(.*?)"', "\n".join(xml_data.split("\n")[:2]))
        assert len(genre) == 1
        genre = genre[0]
        # don't feed the sentencer our pos and lemma predictions, if we have them
        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)
        split_indices = self.predict(no_pos_lemma)

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
            elif any(f"<{elt}>" in line for elt in BLOCK_TAGS) or any(
                f"</{elt}>" in line for elt in BLOCK_TAGS
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
            if line == "<s>":
                in_sent = True
                if len(tokens) > 0:
                    conllu_sentences.append(tokens2conllu(tokens))
                    tokens = []
            elif line == "</s>":
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

        self.load_model()
        # Identify a function that takes data and returns output at the document level
        processing_function = self.split

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function, multithreaded=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("file",help="TT SGML file to test sentence splitting on, or training dir")
    p.add_argument("-m","--mode",choices=["test","train"],default="test")
    p.add_argument(
        "-o",
        "--out_format",
        choices=["binary", "sgml"],
        help="output list of binary split indices or TT SGML",
        default="sgml",
    )

    opts = p.parse_args()
    sentencer = FlairSplitter(config={"LIB_DIR": lib_dir})
    if opts.mode == "train":
        sentencer.train()
    else:
        sgml = io.open(opts.file, encoding="utf8").read()
        result = sentencer.predict(sgml, outmode=opts.out_format)
        print(result)
