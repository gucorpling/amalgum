from .model import ParsingModel
from .tree import RSTTree
from .docreader import DocReader
from .evaluation import Metrics
from .util import drawrst
from .tree2rs3 import to_rs3
from os import listdir
from os.path import join as joinpath
import io, re, os, sys

PY3 = sys.version_info[0] > 2

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

def parse(pm, doc):
    """ Parse one document using the given parsing model

    :type pm: ParsingModel
    :param pm: an well-trained parsing model

    :type fedus: string
    :param fedus: file name of an document (with segmented EDUs) 
    """
    pred_rst = pm.sr_parse(doc)
    return pred_rst


def writebrackets(fname, brackets):
    """ Write the bracketing results into file
    """
    print('Writing parsing results into file: {}'.format(fname))
    with open(fname, 'w') as fout:
        for item in brackets:
            fout.write(str(item) + '\n')


def write_rs3(docname,doc,pred_rst):
    with io.open(docname + ".rs3",'w',encoding="utf8", newline="\n") as f:
        edus = []
        for edu in sorted([e for e in doc.edudict]):
            tokids = doc.edudict[edu]
            edu = []
            for tok in tokids:
                edu.append(doc.tokendict[tok].word)
            edus.append(" ".join(edu))
        notext = pred_rst.parse()
        for i, edu in enumerate(edus):
            edu = edu.replace("(","-LRB-").replace("[","-LSB-").replace(")","-RRB-").replace("]","-RSB-")
            notext = notext.replace("( EDU " + str(i+1) + " )","["+str(i+1)+": "+ edu + " ]")
        withtext = re.sub(r'(\( (SN|NN|NS))',r'\n\1',notext)
        output = ""
        indent = 0
        max_id = 1000
        node_id = 0
        for c in withtext:
            if c == "(":
                output += indent * "  "
                indent += 1
                node_id = str(max_id) + ": "
                max_id += 1
            else:
                node_id = ""
            if c == ")":
                indent -= 1
            output += c + node_id

        output = to_rs3(output)
        if PY3:
            f.write(output)
        else:
            f.write(unicode(output.decode("utf8")))
        return output


def evalparser_multifile(path, report=False,
               bcvocab=None, draw=True,
               withdp=False, fdpvocab=None, fprojmat=None):
    """ Test the parsing performance

    :type path: string
    :param path: path to the evaluation data

    :type report: boolean
    :param report: whether to report (calculate) the f1 score
    """
    # ----------------------------------------
    # Load the parsing model
    print('Load parsing model ...')
    pm = ParsingModel(withdp=withdp,
        fdpvocab=fdpvocab, fprojmat=fprojmat)
    pm.loadmodel(script_dir + "../models/parsing-model.pickle.gz")
    # ----------------------------------------
    # Evaluation
    met = Metrics(levels=['span','nuclearity','relation'])
    # ----------------------------------------
    # Read all files from the given path
    doclist = [joinpath(path, fname) for fname in listdir(path) if fname.endswith('.merge')]
    for fmerge in doclist:
    # ----------------------------------------
    # Read *.merge file
        dr = DocReader()
        doc = dr.read(fmerge)
    # ----------------------------------------
    # Parsing
        pred_rst = pm.sr_parse(doc, bcvocab)
        # Generate rs3 XML
        docname = "tmp.merge"
        # docname = os.path.basename(fmerge).replace(".merge","")
        write_rs3(docname,doc,pred_rst)
        if draw:
            strtree = pred_rst.parse()
            drawrst(strtree, fmerge.replace(".merge",".ps"))

        # Get brackets from parsing results
        pred_brackets = pred_rst.bracketing()
        fbrackets = fmerge.replace('.merge', '.brackets')
        #     # Write brackets into file
        #     writebrackets(fbrackets, pred_brackets)
        #     # ----------------------------------------
        # Evaluate with gold RST tree
        if report:
            fdis = fmerge.replace('.merge', '.dis')
            gold_rst = RSTTree(fdis, fmerge)
            gold_rst.build()
            gold_brackets = gold_rst.bracketing()
            met.eval(gold_rst, pred_rst)

    if report:
        met.report()


def evalparser(fmerge, report=False,
               bcvocab=None, draw=True,
               withdp=False, fdpvocab=None, fprojmat=None,
               pm=None):
    """ Test the parsing performance

    :type path: string
    :param path: path to the evaluation data

    :type report: boolean
    :param report: whether to report (calculate) the f1 score
    """
    # ----------------------------------------
    # Load the parsing model
    if pm is None:
        print('Load parsing model ...')
        pm = ParsingModel(withdp=withdp,
            fdpvocab=fdpvocab, fprojmat=fprojmat)
        pm.loadmodel("lib/dplp_plusplus/models/parsing-model.pickle.gz")
    # ----------------------------------------
    # Evaluation
    met = Metrics(levels=['span','nuclearity','relation'])
    # ----------------------------------------
    # Read all files from the given path
    # doclist = [joinpath(path, fname) for fname in listdir(path) if fname.endswith('.merge')]
    # for fmerge in doclist:
    # ----------------------------------------
    # Read *.merge file
    dr = DocReader()
    doc = dr.read(fmerge)
    # ----------------------------------------
    # Parsing
    pred_rst = pm.sr_parse(doc, bcvocab)
    # Generate rs3 XML
    docname = "tmp.merge"
    # docname = os.path.basename(fmerge).replace(".merge","")
    write_rs3(docname,doc,pred_rst)
    #     if draw:
    #         strtree = pred_rst.parse()
    #         drawrst(strtree, fmerge.replace(".merge",".ps"))
    #     # Get brackets from parsing results
    #     pred_brackets = pred_rst.bracketing()
    #     fbrackets = fmerge.replace('.merge', '.brackets')
    #     # Write brackets into file
    #     writebrackets(fbrackets, pred_brackets)
    #     # ----------------------------------------
    #     # Evaluate with gold RST tree
    #     if report:
    #         fdis = fmerge.replace('.merge', '.dis')
    #         gold_rst = RSTTree(fdis, fmerge)
    #         gold_rst.build()
    #         gold_brackets = gold_rst.bracketing()
    #         met.eval(gold_rst, pred_rst)
    # if report:
    #     met.report()
    return write_rs3(docname,doc,pred_rst)
