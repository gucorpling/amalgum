# -*- coding: utf-8 -*-

"""
nlp_controller.py
A general purpose interface to add annotations to data coming from all genres
"""
import logging
import os, io, sys, re, shutil
from argparse import ArgumentParser
from glob import glob
from time import sleep

from nlp_modules.marmot_tagger import MarmotTagger
from nlp_modules.snlp_parser import StanfordNLPParser
from nlp_modules.tt_tagger import TreeTaggerTagger
from nlp_modules.tt_tokenizer import TreeTaggerTokenizer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
LIB_DIR = SCRIPT_DIR + "lib" + os.sep
BIN_DIR = SCRIPT_DIR + "bin" + os.sep
TT_PATH = BIN_DIR + "treetagger" + os.sep + "bin" + os.sep

MODULES = {
    "tt_tokenizer": TreeTaggerTokenizer,
    "tt_tagger": TreeTaggerTagger,
    "marmot_tagger": MarmotTagger,
    "snlp_parser": StanfordNLPParser,
}


class NLPController:
    def __init__(self, opts):
        logging.info("Initializing NLP Controller...")
        opts.update(
            {
                "SCRIPT_DIR": SCRIPT_DIR,
                "LIB_DIR": LIB_DIR,
                "BIN_DIR": BIN_DIR,
                "TT_PATH": TT_PATH,
            }
        )
        self.opts = opts

        logging.info("Initializing NLP modules...")
        module_slugs = opts["modules"]
        self.pipeline = [MODULES[slug](opts) for slug in module_slugs]
        self.input_dir = opts["input_dir"]
        self.output_dir = opts["output_dir"]

        logging.info("Resolving pipeline module dependencies...")
        satisfied = set()
        for module in self.pipeline:
            if any(req not in satisfied for req in module.__class__.requires):
                formatted_reqs = [
                    f"{m.__class__.__name__}\t"
                    + "{"
                    + ",".join(str(dep) for dep in m.requires)
                    + " -> "
                    + ",".join(str(dep) for dep in m.provides)
                    + "}"
                    for m in self.pipeline
                ]
                raise Exception(
                    f"Invalid pipeline: module {module.__class__} requires "
                    f"{module.__class__.requires}, but pipeline only provides "
                    f"{satisfied}. Full pipeline requirements:\n"
                    + "\n".join(formatted_reqs)
                )
            satisfied.update(module.provides)

        for module in self.pipeline:
            logging.info(
                f"Checking dependencies for module {module.__class__.__name__}..."
            )
            module.test_dependencies()
        # TODO: add dependencies information to each module and verify that this is a valid pipeline

        logging.info("NLPController initialization complete.\n")

    def _init_output_dir(self, initial_dir_path):
        os.makedirs(os.path.join(initial_dir_path, "xml"), exist_ok=True)

        filepaths = glob(os.path.join(self.input_dir, "**/*.xml"), recursive=True)
        logging.info(f"Copying {len(filepaths)} files into {initial_dir_path}...")
        for filepath in filepaths:
            new_filepath = os.path.join(
                initial_dir_path, "xml", filepath.split(os.sep)[-1]
            )
            shutil.copy(filepath, new_filepath)
        logging.info(f"Done copying initial files.\n")

    def run(self):
        last_dir_name = os.path.join(self.output_dir, "00_initial")
        self._init_output_dir(last_dir_name)
        for i, module in enumerate(self.pipeline):
            input_dir = last_dir_name
            output_dir = os.path.join(
                self.output_dir, str(i + 1).zfill(2) + "_" + module.__class__.__name__
            )
            shutil.copytree(input_dir, output_dir)
            logging.info(f"Created directory {output_dir} from {input_dir}.")
            logging.info(f"Running module {module.__class__.__name__}")
            module.run(input_dir, output_dir)
            last_dir_name = output_dir


#        # the files we need to process
#        INPUT_FILE_DIR = SCRIPT_DIR + "out" + os.sep + genre + os.sep + "autogum_*.xml"
#        filepaths = sorted(glob(INPUT_FILE_DIR))
#
#        for file_num, filepath in enumerate(filepaths):
#            with open(filepath, "r", encoding="utf8") as f:
#                raw_xml = f.read()
#            tokenized = tokenize(raw_xml)
#
#            tok_count = len(
#                [t for t in tokenized.strip().split("\n") if not t.startswith("<")]
#            )
#
#            sys.stderr.write(
#                "o Processing document: "
#                + os.path.basename(filepath)
#                + " ("
#                + str(file_num + 1)
#                + "/"
#                + str(len(filepaths))
#                + ")"
#            )
#
#            # Skip documents that are way too big or small
#            if tok_count < 300 or tok_count > 2000:
#                sys.stderr.write(" [skipped due to size]\n")
#                continue
#            else:
#                sys.stderr.write("\n")
#
#            # POS tag
#            # If we want to tag outside StanfordNLP, a dedicated tagger can be used
#            if opts.no_parse:
#                tagged = pos_tag(tokenized)
#
#            # Add sentence splits - note this currently produces mal-nested SGML
#            if opts.no_sent:
#                split_indices = [1] + [0] * (len(tokenized) - 1)
#            else:
#                from lib.gumdrop.EnsembleSentencer import EnsembleSentencer
#
#                best_sentencer_ever = EnsembleSentencer(
#                    lang="eng", model="eng.rst.gum", genre_pat="_([^_]+)_"
#                )
#                split_indices = best_sentencer_ever.predict(
#                    tokenized, as_text=True, plain=True, genre=genre
#                )
#
#            counter = 0
#            splitted = []
#            opened_sent = False
#            para = True
#            for line in tokenized.strip().split("\n"):
#                if not (line.startswith("<") and line.endswith(">")):
#                    # Token
#                    if split_indices[counter] == 1 or para:
#                        if opened_sent:
#                            splitted.append("</s>")
#                            opened_sent = False
#                        splitted.append("<s>")
#                        opened_sent = True
#                        para = False
#                    counter += 1
#                elif (
#                        "<p>" in line or "<head>" in line or "<caption>" in line
#                ):  # New block, force sentence split
#                    para = True
#                splitted.append(line)
#            splitted = "\n".join(splitted)
#            if opened_sent:
#                if splitted.endswith("</text>"):
#                    splitted = splitted.replace("</text>", "</s>\n</text>")
#                else:
#                    splitted += "\n</s>"
#
#            if not opts.no_parse:
#                # Parse
#                no_xml = splitted.replace("</s>\n<s>", "---SENT---")
#                no_xml = re.sub(r"<[^<>]+>\n?", "", no_xml)
#
#                sents = no_xml.strip().replace("\n", " ").replace("---SENT--- ", "\n")
#                parsed = dep_parse(sents, snlp, torch)
#            else:
#                parsed = tagged
#
#            doc = os.path.basename(filepath)
#
#            # Insert tags into XML
#            pos_lines = []
#            lemma_lines = []
#            for line in parsed.split("\n"):
#                if "\t" in line:
#                    fields = line.split("\t")
#                    if opts.no_parse:
#                        lemma, xpos = fields[2], fields[1]
#                    else:
#                        lemma, xpos = fields[2], fields[4]
#                    pos_lines.append(xpos)
#                    lemma_lines.append(lemma)
#            tagged = []
#            counter = 0
#            for line in splitted.split("\n"):
#                if line.startswith("<") and line.endswith(">"):
#                    tagged.append(line)
#                else:
#                    line = line + "\t" + pos_lines[counter] + "\t" + lemma_lines[counter]
#                    tagged.append(line)
#                    counter += 1
#            tagged = "\n".join(tagged)
#
#            # Write output files
#            with io.open(XML_OUTPUT_DIR + doc, "w", encoding="utf8", newline="\n") as f:
#                f.write(tagged)
#
#            if not opts.no_parse:
#                with io.open(
#                        DEPENDENCY_OUTPUT_DIR + doc.replace(".xml", ".conllu"),
#                        "w",
#                        encoding="utf8",
#                        newline="\n",
#                ) as f:
#                    f.write(parsed)


def main():
    p = ArgumentParser()
    p.add_argument("output_dir", help="The directory that output should be written to.")
    p.add_argument(
        "-m",
        "--modules",
        nargs="+",
        choices=MODULES.keys(),
        default=["tt_tokenizer", "tt_tagger"],
        help="NLP pipeline modules, included in the order they are specified.",
    )
    p.add_argument(
        "-i",
        "--input-dir",
        default="out",
        help="The directory that holds the unprocessed XML files. Useful for prototyping on a small set of documents.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "By default, the pipeline will refuse to run if the output directory "
            "already exists. Setting this flag will REMOVE all the data in the "
            "output directory before new data is introduced."
        ),
    )
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="Modules will attempt to use GPU if this flag is provided.",
    )
    opts = p.parse_args()

    # Check if output directory already exists.
    if os.path.exists(opts.output_dir):
        if opts.overwrite:
            logging.warning(
                f"About to delete ALL data from {opts.output_dir}. Interrupt now if you want to keep it."
            )
            for i in range(5, 0, -1):
                print(f"Deleting in {i}s...\r", end="")
                sleep(1)
            shutil.rmtree(opts.output_dir)
            os.mkdir(opts.output_dir)
            logging.info(f"Deleted and re-created {opts.output_dir}.\n")
        else:
            raise Exception(
                "Output path "
                + opts.output_dir
                + " already exists. Use the flag --overwrite "
                "if you know this and want to LOSE all the data in it."
            )

    controller = NLPController(vars(opts))
    controller.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
