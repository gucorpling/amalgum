import os
import platform
import sys
from os import environ as env
from xml.dom import minidom

from lib.utils import get_col, exec_via_temp
from nlp_modules.base import NLPModule, NLPDependencyException, PipelineDep


class MarmotTagger(NLPModule):
    requires = (PipelineDep.TOKENIZE,)
    provides = (PipelineDep.POS_TAG,)

    def __init__(self, config):
        jar_sep = ";" if platform.system() == "Windows" else ":"
        self.command = [
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

    def test_dependencies(self):
        if not os.path.exists("marmot.jar"):
            raise NLPDependencyException(
                "Could not locate file `marmot.jar`. Please download marmot:"
                "http://cistern.cis.lmu.de/marmot/bin/CURRENT/"
            )
        if not os.path.exists("trove.jar"):
            raise NLPDependencyException(
                "Could not locate file `trove.jar`. Please download trove.jar:"
                "http://cistern.cis.lmu.de/marmot/bin/CURRENT/"
            )

    def run(self, input_dir, output_dir):
        processing_function = self.tag
        self.process_files(input_dir, output_dir, processing_function)

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
