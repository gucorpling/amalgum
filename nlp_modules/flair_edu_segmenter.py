from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    TransformerWordEmbeddings
)
from flair.models import SequenceTagger
import flair

import os, re, io

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
lib_dir = script_dir + ".." + os.sep + "lib" + os.sep
nlp_modules_dir = script_dir + ".." + os.sep + "nlp_modules" + os.sep
flair_splitter_dep_dir = nlp_modules_dir + "splitter-dependencies" + os.sep

try:
    from nlp_modules.base import NLPModule, PipelineDep
except:
    from base import NLPModule, PipelineDep
from collections import OrderedDict, defaultdict


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def unescape(token):
    token = token.replace("&quot;", '"')
    token = token.replace("&lt;", "<")
    token = token.replace("&gt;", ">")
    token = token.replace("&amp;", "&")
    token = token.replace("&apos;", "'")
    return token


class FlairEDUSplitter(NLPModule):
    requires = (PipelineDep.TOKENIZE,)
    provides = (PipelineDep.EDUS,)

    def __init__(self, config, model_path=None, span_size=6):

        # Number of tokens to include as pre/post context around each sentence
        self.span_size = span_size
        # Numerical stride size only needed for sent mode; EDU mode strides by sentences
        self.LIB_DIR = config["LIB_DIR"]
        self.test_dependencies()
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.model = None

    def load_model(self, path=None):

        model = "flair-splitter-edu.pt"
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
        model = "flair-splitter-edu.pt"
        if not os.path.exists(flair_splitter_dep_dir + model):
            self.download_file(model, flair_splitter_dep_dir, subfolder="split")

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
            data_folder, columns, train_file="edu_train.txt", test_file="edu_test.txt", dev_file="edu_dev.txt",
        )

        print(corpus)

        tag_type = "ner"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)

        # initialize embeddings
        embedding_types = [
            # WordEmbeddings('glove'),
            # comment in this line to use character embeddings
            CharacterEmbeddings(),
            # comment in these lines to use flair embeddings
            #FlairEmbeddings("news-forward"),
            #FlairEmbeddings("news-backward"),
            TransformerWordEmbeddings("google/electra-base-discriminator")
            # BertEmbeddings('distilbert-base-cased')
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=128, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=False, use_rnn=True
        )

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        trainer.train(training_dir, learning_rate=0.1, mini_batch_size=16, max_epochs=30)
        self.model = tagger

    def predict(self, tt_sgml, outmode="binary"):

        def is_tok(sgml_line):
            return len(sgml_line) > 0 and not (sgml_line.startswith("<") and sgml_line.endswith(">"))

        def is_sent(line):
            return line in ["<s>", "</s>"] or line.startswith("<s ")

        if self.model is None:
            self.load_model()

        final_mapping = {}  # Map each contextualized token to its (sequence_number, position)
        spans = []  # Holds flair Sentence objects for labeling

        # Do EDU segmentation, input TT SGML has sentence tags like '\n</s>\n'
        sents = tt_sgml.split("</s>")[:-1]
        sent_tokens = defaultdict(list)
        for i, sent in enumerate(sents):
            toks = []
            for line in sent.split("\n"):
                if is_tok(line):
                    tok = line.split("\t")[0]
                    toks.append(tok)
            sent_tokens[i] = toks

        counter = 0
        for i, sent in enumerate(sents):
            span = []
            # Use last sent as prev if this is sent 1
            prev_s = sent_tokens[i - 1] if i > 0 else sent_tokens[len(sents) - 1]
            pre_context = prev_s[-self.span_size :] if len(prev_s) > 5 else prev_s[:]
            pre_context.append("<pre>")
            # Use first sent as next if this is last sent
            next_s = sent_tokens[i + 1] if i < len(sents) - 1 else sent_tokens[0]
            post_context = next_s[: self.span_size] if len(next_s) > 5 else next_s[:]
            post_context = ["<post>"] + post_context

            for tok in pre_context:
                span.append(tok)
            for j, tok in enumerate(sent_tokens[i]):
                span.append(tok)
                # The aligned prediction will be in Sentence i, at the position
                # after span_size + 1 (pre-context + <pre> token) + counter
                final_mapping[counter] = (i, len(pre_context) + j)
                counter += 1
            for tok in post_context:
                span.append(tok)
            spans.append(Sentence(" ".join(span), use_tokenizer=lambda x: x.split()))

        # Predict
        preds = self.model.predict(spans)

        if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
            preds = spans

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

        # Generate rs3 style <segment> elements
        counter = 0
        edu_num = 0
        output = ""
        first = True
        buffer = []
        for line in tt_sgml.strip().split("\n"):
            if is_sent(line) and counter < len(labels):
                labels[counter] = 1  # Force EDU break on sentence break
            elif is_tok(line):
                if labels[counter] == 1:
                    if first:
                        edu_num += 1
                        output += '<segment id="'+str(edu_num)+'">'
                        first = False
                    if len(buffer) > 0:
                        edu_num += 1
                        output += " ".join(buffer) + '</segment>\n<segment id="'+str(edu_num)+'">'
                    buffer = []
                buffer.append(line.split("\t")[0])
                counter += 1
        if len(buffer) > 0:
            output += " ".join(buffer) + "</segment>"
        output = re.sub(r"\n<segment[^<>\n]+>$", "", output)

        return output.strip() + "\n"

    def split(self, context):
        xml_data = context["xml"]
        # Sometimes the tokenizer doesn't newline every elt
        xml_data = xml_data.replace("><", ">\n<")
        # Ad hoc fix for a tokenization error
        xml_data = xml_data.replace("°<", "°\n<")
        # Remove empty elements?
        # for elt in TAGS:
        #    xml_data = xml_data.replace(f"<{elt}>\n</{elt}>\n", "")

        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)

        seg_output = self.predict(no_pos_lemma, outmode="edus")
        in_toks = len([l for l in xml_data.strip().split("\n") if not is_sgml_tag(l)])

        # Check we're returning as many tokens as we got
        segs_text = re.sub(r'</?segment[^<>]+>','',seg_output).strip().replace('\n',' ')
        segs_toks = segs_text.count(" ") + 1
        assert segs_toks == in_toks

        return {"rst": seg_output}

    def run(self, input_dir, output_dir):

        self.load_model()
        # Identify a function that takes data and returns output at the document level
        processing_function = self.split

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function, multithreaded=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("file", help="TT SGML file to test sentence splitting on, or training dir")
    p.add_argument("-m", "--mode", choices=["test", "train"], default="test")
    p.add_argument(
        "-o",
        "--out_format",
        choices=["binary", "edus"],
        help="output list of binary split indices or TT SGML",
        default="edus",
    )

    opts = p.parse_args()
    splitter = FlairEDUSplitter(config={"LIB_DIR": lib_dir})
    if opts.mode == "train":
        splitter.train()
    else:
        sgml = io.open(opts.file, encoding="utf8").read()
        result = splitter.predict(sgml, outmode=opts.out_format)
        print(result)
