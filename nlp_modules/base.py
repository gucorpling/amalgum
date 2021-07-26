import logging
import os, io, sys, requests, shutil
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
import pmap
from functools import partial

from tqdm import tqdm


class PipelineDep(Enum):
    TOKENIZE = "TOKENIZE"  # XML that is tokenized tree-tagger style
    POS_TAG = "POS_TAG"  # POS tags
    S_TYPE = "S_TYPE"  # sentence types
    S_SPLIT = "S_SPLIT"  # sentence splitting
    EDUS = "EDUS"  # EDU segmenting
    PARSE = "PARSE"  # dependency parse
    ACE_ENTITIES = "ACE_ENTITIES"  # nested entity prediction in ACE format

    # only produced
    RST_OUT = "RST_OUT"  # output for rhetorical structure theory
    TSV_OUT = "TSV_OUT"  # coref/entity output

    DATETIME = "DATETIME"

    def __str__(self):
        return self.value


class NLPModule(ABC):
    """Superclass for all NLP modules, which are elements of the NLP pipeline."""

    @abstractmethod
    def __init__(self, config):
        """Config is a dict containing command-line arguments and other information.
        See nlp_controller.py."""
        pass

    @property
    @abstractmethod
    def requires(self):
        """Returns a tuple of PipelineDeps that must be provided before this step of the pipeline.
        For instance, a POS tagger should return `(PipelineDep.TOKENIZE,)` here since it requires
        tokens to do its work."""
        return ()

    @property
    @abstractmethod
    def provides(self):
        """Returns a tuple of PipelineDeps that this module provides for downstream modules.
        For instance, a POS tagger should return `(PipelineDep.POS_TAG,)` here since it provides
        POS tags for any downstream modules that might need them (like a dependency parser)
        """
        return ()

    @abstractmethod
    def test_dependencies(self):
        """
        You MUST use this method to load any dependencies that will be required for your NLP module.
        This method will be called when the full pipeline is initialized to make sure that all required
        libraries, binaries, models, etc. are present. If this method runs without incident, then
        it is assumed that all dependencies are present. If there is a dependency issue, this method
        should raise an NLPDependencyException.

        It is also acceptable to use this function to initialize dependencies if they are not prepared,
        as long as the behavior specified above is preserved. (StanfordNLPParser, for instance,
        downloads its models in test_dependencies.)

        If no dependencies are needed, implement this method and have its body just be `pass`.
        """
        pass

    @abstractmethod
    def run(self, input_dir, output_dir):
        """
        Top-level function for document processing that is invoked by the NLP controller. Your module
        should NOT use hard-coded file directories, but rather should rely on the values of `input_dir`
        and `output_dir` from this function. In return the NLP controller will guarantee that
        (1) XML versions of the documents will be available at `{input_dir,output_dir}/xml/`
        (2) Other versions of the documents will be available at `{input_dir,output_dir}/{dep,xml,...}/`
        Note that document names will no longer be separated by a folder indicating its genre. (This
        information is already present in the filename.)
        :param input_dir: The base directory for files to be used in this module. Files in here SHOULD
                          NOT be modified.
        :param output_dir: The base directory where files will be written by this module. Files should
                           have a path that is IDENTICAL to their paths in `input_dir`.
        :return: None
        """
        pass

    def process_files(
        self,
        input_dir,
        output_dir,
        process_document_content,
        file_type="xml",
        multithreaded=False,
    ):
        """
        Handles the most common case of iteration where processing can be handled with a function that is
        applied to the contents of each file. (Not every module can accommodate this, e.g. modules that need
        to write to multiple directories.) This allows the processing function to ignore file I/O.
        :param input_dir: From `run`
        :param output_dir: From `run`
        :param process_document_content: A method that accepts a single argument, the contents of an input file.
        :param file_type: The AMALGUM file type folder that should be used under input_dir and output_dir.
        :param multithreaded: Use python-pmap to apply the process_document_content_dict in parallel. Only use if
                              the function is CPU-intensive (and not, e.g., I/O intensive) and there are no race
                              conditions that would be introduced by having multiple threads use the function.
                              (If in doubt, don't set this to True!)
        :return: None
        """
        os.makedirs(os.path.join(output_dir, file_type), exist_ok=True)
        sorted_filepaths = sorted(glob(os.path.join(input_dir, file_type, "*")))

        progress = tqdm(total=len(sorted_filepaths))

        def process_file(filepath, report_progress=False):
            nonlocal progress

            filename = filepath.split(os.sep)[-1]
            with io.open(filepath, "r", encoding="utf8") as f:
                s = f.read()
            try:
                s = process_document_content(s)
            except Exception as e:
                logging.error(f"Encountered an error while processing file {filepath}!")
                raise e
            with io.open(
                os.path.join(output_dir, file_type, filename),
                "w",
                encoding="utf8",
                newline="\n",
            ) as f:
                f.write(s)

            # This could lead to race conditions, but it doesn't really matter if
            # the count gets messed up, so we let it be
            if report_progress:
                progress.update(1)

        if multithreaded:
            list(
                pmap.pmap(partial(process_file, report_progress=True), sorted_filepaths)
            )
            progress.close()
        else:
            for filepath in tqdm(sorted_filepaths):
                process_file(filepath)

    # A map from the subdirectory name to the extension of the files that go in that dir.
    FILE_EXT_MAP = {"rst": "rs3", "dep": "conllu"}

    def process_files_multiformat(
        self, input_dir, output_dir, process_document_content_dict, multithreaded=False
    ):
        """
        Like process_files, with one difference: the supplied function `process_document_content_dict` now
        (1) receives a dict of dir -> file contents, e.g. {'xml': '<text ...>...</text>', 'rst': '...', ...},
            which contains every version of the document that is currently in the pipeline
        (2) expects a dict with the same structure to be returned, e.g. {'tsv': '...'}. Every pair in the
            returned dict will be written to the appropriate file, e.g. 'tsv/doc_name.tsv', 'rst/doc_name.rs3'.
            File extension is determined from NLPModule.FILE_EXT_MAP, or if it is not present there, is assumed
            to be the same as the name of the subdirectory.
        :param input_dir: From `run`
        :param output_dir: From `run`
        :param process_document_content_dict: A method that accepts a single argument, a dict with
                                              the key being the subdirectory that the file is in, and
                                              the value being the contents of that file as a string
        :param output_dir: if True, use the python-pmap library to run the document processing function in parallel.
                           Do NOT set this to True unless you are CERTAIN that there will not be any race conditions
                           that could corrupt the data.
        :param multithreaded: Use python-pmap to apply the process_document_content_dict in parallel. Only use if
                              the function is CPU-intensive (and not, e.g., I/O intensive) and there are no race
                              conditions that would be introduced by having multiple threads use the function.
                              (If in doubt, don't set this to True!)
        :return: None
        """
        existing_input_dirs = [
            os.path.join(input_dir, subdir)
            for subdir in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, subdir))
        ]
        if len(existing_input_dirs) == 0:
            raise Exception("No input directories found!")

        # Use the first dir to derive filenames without filetypes
        base_dir = sorted(existing_input_dirs)[0]
        filenames = sorted(
            [filename.split(".")[0] for filename in os.listdir(base_dir)]
        )

        progress = tqdm(total=len(filenames))

        def process_filename(filename, report_progress=False):
            nonlocal progress
            # Refuse to proceed if every other directory doesn't also have a file with the same name
            if not all(
                any(fname.startswith(filename) for fname in os.listdir(subdir))
                for subdir in existing_input_dirs
            ):
                raise Exception(
                    f"File {filename} does not exist in all of these directories: {existing_input_dirs}"
                )

            # construct the content dict
            content_dict = {}
            content_dict["filename"] = filename
            for subdir in existing_input_dirs:
                matching_files = [
                    f for f in os.listdir(subdir) if f.split(".")[0] == filename
                ]
                assert (
                    len(matching_files) > 0
                ), f"Couldn't find {filename} in directory {subdir}"
                assert (
                    len(matching_files) < 2
                ), f"More than one file starting with {filename} in directory {subdir}"

                filepath = os.path.join(subdir, matching_files[0])
                with io.open(filepath, "r", encoding="utf8") as f:
                    content_dict[subdir.split(os.sep)[-1]] = f.read()

            # run the processing function
            try:
                output_dict = process_document_content_dict(content_dict)
            except Exception as e:
                logging.error(f"Encountered an error while processing file {filepath}!")
                raise e

            # write out all the output documents
            for subdir, content in output_dict.items():
                subdir_path = os.path.join(output_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path)

                file_ext = (
                    NLPModule.FILE_EXT_MAP[subdir]
                    if subdir in NLPModule.FILE_EXT_MAP
                    else subdir
                )
                filepath = os.path.join(output_dir, subdir, filename + "." + file_ext)
                with io.open(filepath, "w", encoding="utf8", newline="\n") as f:
                    f.write(content)

            # This could lead to race conditions, but it doesn't really matter if
            # the count gets messed up, so we let it be
            if report_progress:
                progress.update(1)

        if multithreaded:
            list(pmap.pmap(partial(process_filename, report_progress=True), filenames))
            progress.close()
        else:
            for filename in tqdm(filenames):
                process_filename(filename)

    @staticmethod
    def download_file(model, local_path, subfolder=None):
        server = "corpling.uis.georgetown.edu"
        resource = [server, "amir", "download", "amalgum"]
        if subfolder is not None:
            resource.append(subfolder)
        resource.append(model)
        resource = "/".join(resource)
        url = "https://" + resource
        if not local_path.endswith(model):
            if not local_path.endswith(os.sep):
                local_path += os.sep
            local_path += model
        try:
            sys.stderr.write("o Downloading model from " + url + "...\n")
            with requests.get(url, stream=True) as r:
                write_dir = os.sep.join(local_path.split(os.sep)[:-1])
                if not os.path.exists(write_dir):
                    os.makedirs(write_dir)
                with io.open(local_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            sys.stderr.write("o Download successful\n")
        except Exception as e:
            sys.stderr.write("\n! Could not download model from " + url + "\n")
            sys.stderr.write(str(e))
            raise NLPDependencyException()


class NLPDependencyException(Exception):
    """Raise in case a dependency wasn't satisfied in test_dependencies"""
