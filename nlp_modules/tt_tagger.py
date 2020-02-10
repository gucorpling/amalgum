import os
import sys
from xml.dom import minidom

from lib.utils import exec_via_temp, get_col
from nlp_modules.base import NLPModule


class TreeTaggerTagger(NLPModule):
    def __init__(self, config):
        TT_PATH = config["TT_PATH"]
        self.TT_PATH = TT_PATH
        self.tagger_bin = TT_PATH + "tree-tagger"
        self.english_par = TT_PATH + "english.par"
        self.command = [
            self.tagger_bin,
            self.english_par,
            "-lemma",
            "-no-unknown",
            "-sgml",
            "tempfilename",
        ]

    def test_dependencies(self):
        if not os.path.exists(self.TT_PATH + "tree-tagger"):
            sys.stderr.write(
                "TreeTagger binary not available at " + self.TT_PATH + "tree-tagger\n"
            )
            sys.stderr.write(
                "Download TreeTagger binary from https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/\n"
            )
            sys.exit(1)
        if not os.path.exists(self.TT_PATH + "english.par"):
            sys.stderr.write(
                "TreeTagger English parameter file not found at "
                + self.TT_PATH
                + "english.par\n"
            )
            sys.stderr.write(
                "Download English PTB parameter file from https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz\n"
            )
            sys.exit(1)

    def tag(self, tokenized_document):
        data = tokenized_document.strip().split("\n")
        orig_data_split = data
        data = [
            word.strip().replace("’", "'") for word in data
        ]  # Use plain quotes for easier tagging
        data = "\n".join(data)

        tagged = exec_via_temp(data, self.command, outfile=False)
        tagged = tagged.strip().replace("\r", "").split("\n")
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

    def run(self, input_dir, output_dir):
        processing_function = self.tag
        self.process_files(input_dir, output_dir, processing_function)
