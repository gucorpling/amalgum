import logging
import os
from abc import ABC, abstractmethod
from glob import glob
from traceback import print_exc

from tqdm import tqdm


class NLPModule(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

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
        self, input_dir, output_dir, process_document_content, file_type="xml"
    ):
        """
        Handles the most common case of iteration where processing can be handled with a function that is
        applied to the contents of each file. (Not every module can accommodate this, e.g. modules that need
        to write to multiple directories.) This allows the processing function to ignore file I/O.
        :param input_dir: From `run`
        :param output_dir: From `run`
        :param process_document_content: A method that accepts a single argument, the contents of an input file.
        :param file_type: The AMALGUM file type folder that should be used under input_dir and output_dir.
        :return: None
        """
        os.makedirs(os.path.join(output_dir, file_type), exist_ok=True)
        sorted_filepaths = sorted(glob(os.path.join(input_dir, file_type, "*")))
        for filepath in tqdm(sorted_filepaths):
            filename = filepath.split(os.sep)[-1]
            with open(filepath, "r") as f:
                s = f.read()
            try:
                s = process_document_content(s)
            except Exception as e:
                logging.error(f"Encountered an error while processing file {filepath}!")
                raise e
            with open(os.path.join(output_dir, file_type, filename), "w") as f:
                f.write(s)


class NLPDependencyException(Exception):
    """Raise in case a dependency wasn't satisfied in test_dependencies"""
