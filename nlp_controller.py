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
from nlp_modules.gumdrop_splitter import GumdropSplitter

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
LIB_DIR = SCRIPT_DIR + "lib" + os.sep
BIN_DIR = SCRIPT_DIR + "bin" + os.sep
TT_PATH = BIN_DIR + "treetagger" + os.sep + "bin" + os.sep

MODULES = {
    "tt_tokenizer": TreeTaggerTokenizer,
    "tt_tagger": TreeTaggerTagger,
    "marmot_tagger": MarmotTagger,
    "snlp_parser": StanfordNLPParser,
    "gumdrop_splitter": GumdropSplitter,
}


def lazy_pipeline(slugs, opts):
    for slug in slugs:
        yield MODULES[slug](opts)


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

        logging.info("NLPController initialization complete.\n")

    def _init_output_dir(self, initial_dir_path):
        """Copy the input directory into the output directory."""
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
        """Create the output directory and run every step of the pipeline in sequence,
        creating a fresh directory for each step."""
        begin_step = self.opts["begin_step"]

        # init dirs if we're not skipping a step
        if begin_step == 0:
            last_dir_name = os.path.join(self.output_dir, "00_initial")
            self._init_output_dir(last_dir_name)
        # if we are skipping steps, delete the dirs after the skipped steps
        else:
            last_dir_name = glob(
                os.path.join(self.output_dir, str(begin_step).zfill(2) + "*")
            )[0]
            dirs_to_delete = [
                glob(os.path.join(self.output_dir, str(i).zfill(2) + "*"))[0]
                for i in range(begin_step + 1, len(self.pipeline) + 1)
            ]
            for dirname in dirs_to_delete:
                print(f"removing {dirname}")
                shutil.rmtree(dirname)

        steps = list(enumerate(self.pipeline))[self.opts["begin_step"] :]
        for i, module in steps:
            input_dir = last_dir_name
            output_dir = os.path.join(
                self.output_dir, str(i + 1).zfill(2) + "_" + module.__class__.__name__
            )
            shutil.copytree(input_dir, output_dir)
            logging.info(f"Created directory {output_dir} from {input_dir}.")
            logging.info(f"Running module {module.__class__.__name__}")
            module.run(input_dir, output_dir)
            last_dir_name = output_dir


def main():
    p = ArgumentParser()
    p.add_argument("output_dir", help="The directory that output should be written to.")
    p.add_argument(
        "-m",
        "--modules",
        nargs="+",
        choices=MODULES.keys(),
        default=["tt_tokenizer", "tt_tagger", "gumdrop_splitter"],
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
    p.add_argument(
        "--begin-step",
        type=int,
        default=0,
        help=(
            "If provided, will begin the pipeline from the ZERO-INDEXED step "
            "in the pipeline corresponding to this value. E.g., --resume-step 1"
            " would resume step y in the pipeline [x y z]. Every other step "
            "before the resumed step is assumed to have been executed successfully."
            " If the value if this parameter is greater than 0, --overwrite is ignored."
        ),
    )
    opts = p.parse_args()

    # Check if output directory already exists.
    if os.path.exists(opts.output_dir):
        if opts.overwrite and opts.begin_step == 0:
            logging.warning(
                f"About to delete ALL data from {opts.output_dir}. Interrupt now if you want to keep it."
            )
            for i in range(3, 0, -1):
                print(f"Deleting in {i}s...\r", end="")
                sleep(1)
            shutil.rmtree(opts.output_dir)
            os.mkdir(opts.output_dir)
            logging.info(f"Deleted and re-created {opts.output_dir}.\n")
        elif opts.begin_step == 0:
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
