import numpy as np
from random import seed
from argparse import ArgumentParser
import gzip, sys, os
from src.readdoc import readdoc
from src.data import Data
from src.model import ParsingModel
from src.util import reversedict
from src.evalparser import evalparser, evalparser_multifile
from glob import glob

PY3 = sys.version_info[0] > 2

if PY3:
    from pickle import load
else:
    from cPickle import load

seed(42)
np.random.seed(42)

WITHDP = False

def createdoc():
    ftrn = "data/sample/trn-doc.pickle.gz"
    rpath = "data/training/"
    readdoc(rpath, ftrn)
    ftst = "data/sample/tst-doc.pickle.gz"
    rpath = "data/test/"
    readdoc(rpath, ftst)


def createtrndata(path="data/train/", topn=8000, bcvocab=None):
    data = Data(bcvocab=bcvocab,
                withdp=WITHDP,
                fdpvocab="data/resources/word-dict.pickle.gz",
                fprojmat="data/resources/projmat.pickle.gz")
    data.builddata(path)
    data.buildvocab(topn=topn)
    data.buildmatrix()
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    data.savematrix(fdata, flabel)
    data.savevocab("data/sample/vocab.pickle.gz")


def trainmodel():
    fvocab = "data/sample/vocab.pickle.gz"
    fdata = "data/sample/trn.data"
    flabel = "data/sample/trn.label"
    D = load(gzip.open(fvocab))
    vocab, labelidxmap = D['vocab'], D['labelidxmap']
    print('len(vocab) = {}'.format(len(vocab)))
    data = Data()
    trnM, trnL = data.loadmatrix(fdata, flabel)
    print('trnM.shape = {}'.format(trnM.shape))
    idxlabelmap = reversedict(labelidxmap)
    pm = ParsingModel(vocab=vocab, idxlabelmap=idxlabelmap)
    pm.train(trnM, trnL)
    pm.savemodel("models/parsing-model.pickle.gz")


def run_merge(paths):
    files = glob(paths["dis"] + "*.dis")
    for file_ in files:
        docname = os.path.basename(file_).replace(".dis", "")
        xml = paths["xml"] + docname + ".xml"
        dep = paths["dep"] + docname + ".conllu"
        merge(file_, xml, dep, docname, as_text=False, outdir=paths["dis"])


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("-t","--train", action="store_true", help="train the parser")
    p.add_argument("-n", "--nfeats", default=9000, help="maximum number of features in training")
    p.add_argument("--trainpath", default="data/train/", help="location of training data .dis and .merge files")
    p.add_argument("--testpath", default="data/test/", help="location of test data .dis and .merge  files")
    p.add_argument("--parsepath", default="data/dep/", help="location of parsed .conllu files")
    p.add_argument("--xmlpath", default="data/xml/", help="location of .xml files with document structure")
    p.add_argument("--processed", action="store_true", help="indicates .merge files are already processed")

    opts = p.parse_args()
    paths = {"dis": opts.testpath, "xml":opts.xmlpath, "dep":opts.parsepath}

    bcvocab = None

    # Build .merge files if needed
    overwrite_merge = True
    if not opts.processed:
        from src.make_merge import merge
        if opts.train:
            paths["dis"] = opts.trainpath
            for path in paths:  # Normalize paths for OS
                if "/" in paths[path] and "/" != os.sep:
                    paths[path] = paths[path].replace("/",os.sep)
                if not paths[path].endswith(os.sep):
                    paths[path] += os.sep
            if len(glob(paths["dis"] + "*.merge")) > 0:
                if PY3:
                    prompt = input("\nFound *.merge files in " + paths["dis"] + ". Overwrite? [y/n]").lower()
                else:
                    prompt = raw_input("\nFound *.merge files in " + paths["dis"] + ". Overwrite? [y/n]").lower()
                if not prompt.startswith("y"):
                    overwrite_merge = False

            if overwrite_merge:
                run_merge(paths)

        paths["dis"] = opts.testpath
        if overwrite_merge:
            run_merge(paths)

    # Use brown clsuters
    with gzip.open("resources/bc3200.pickle.gz") as fin:
        print('Load Brown clusters for creating features ...')
        bcvocab = load(fin)
    if opts.train:
        # Create training data
        createtrndata(path="data/train/", topn=opts.nfeats, bcvocab=bcvocab)
        # Train model
        trainmodel()
    # Evaluate model on test set
    evalparser_multifile(path="data/test/", report=True,
               bcvocab=bcvocab, draw=False,
               withdp=WITHDP,
               fdpvocab="data/resources/word-dict.pickle.gz",
               fprojmat="data/resources/projmat.pickle.gz")
