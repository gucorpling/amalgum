import os, io, shutil
import sys
import gzip
import requests
from nlp_modules.base import NLPModule, PipelineDep
from lib.dplp_plusplus.src.docreader import DocReader
from lib.dplp_plusplus.src.make_merge import Sequencer, merge
from lib.dplp_plusplus.src.evalparser import evalparser
from nlp_modules.configuration import MODEL_SERVER,RSTDT_MODEL_PATH

PY3 = sys.version_info[0] > 2

if PY3:
    from pickle import load
else:
    from cPickle import load

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
lib_dir = script_dir + ".." + os.sep + "lib"

class RSTParser(NLPModule):

    requires = (PipelineDep.S_TYPE, PipelineDep.PARSE)
    provides = (PipelineDep.RST_OUT,)

    def __init__(self, opts):
        self.test_dependencies()
        self.sequencer = Sequencer()
        self.dr = DocReader()
        # Use brown clsuters
        with gzip.open(lib_dir + "/dplp_plusplus/resources/bc3200.pickle.gz") as fin:
            print('Load Brown clusters for creating features ...')
            self.bcvocab = load(fin)

    @staticmethod
    def download_file(url, local_path):
        try:
            sys.stderr.write("o Downloading model from " + url + "...\n")
            with requests.get(url, stream=True) as r:
                with io.open(local_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            sys.stderr.write("o Download successful\n")
        except Exception as e:
            sys.stderr.write("\n! Could not download model from " + url + "\n")
            sys.stderr.write(str(e))

    def test_dependencies(self):
        import flair            # version==0.4.4
        import torch            # version==1.2.0; torchvision==0.4.0; tqdm==4.43.0
        # Check flair and torch version
        #if flair.__version__ != '0.4.4':
        #    raise Exception("flair version 0.4.4 must be used with rstParser")
        #if torch.__version__ != '1.2.0':
        #    raise Exception("Torch version 1.4.0 must be used with rstParser")

        # Check the pretrain model
        model_dir = os.path.join(lib_dir, "dplp_plusplus", "models")
        if not os.path.exists(model_dir+os.sep+"rstdt_collapsed.pt"):
            self.download_file(MODEL_SERVER + RSTDT_MODEL_PATH, model_dir+'/rstdt_collapsed.pt')

    def parse(self, doc_dict):

        # construct .merge file that contains all the features
        fmerge = merge(doc_dict["rst"], doc_dict["xml"], doc_dict["dep"], doc_dict["filename"], seq=self.sequencer, as_text=True)

        rst_out = evalparser(fmerge, report=False, bcvocab=self.bcvocab,  draw=False, withdp=False,
                   fdpvocab="data/resources/word-dict.pickle.gz", fprojmat="data/resources/projmat.pickle.gz")

        return {"rst": rst_out}

    def run(self, input_dir, output_dir):

        import sys
        sys.setrecursionlimit(3000)
        processing_function = self.parse

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function, multithreaded=False)


def test_main():
    test_filename = "GUM_academic_art"
    test_xml = """<text id="GUM_academic_art" author="Claire Bailey-Ross, Andrew Beresford, Daniel Smith, Claire Warwick" dateCollected="2017-09-13" dateCreated="2017-08-08" dateModified="2017-09-13" shortTitle="art" sourceURL="https://dh2017.adho.org/abstracts/333/333.pdf" speakerCount="0" speakerList="none" title="Aesthetic Appreciation and Spanish Art: Insights from Eye-Tracking" type="academic">
    <head>
    <hi rend="bold blue">
    <s>
    Aesthetic	JJ	aesthetic
    Appreciation	NN	appreciation
    and	CC	and
    Spanish	JJ	Spanish
    Art	NN	art
    :	:	:
    </s>
    <s>
    Insights	NNS	insight
    from	IN	from
    Eye-Tracking	NN	eye-tracking
    </s>
    </hi>
    </head>"""

    test_conll = """# newdoc id = GUM_academic_art
# sent_id = GUM_academic_art-1
# text = Aesthetic Appreciation and Spanish Art:
# s_type=frag
1	Aesthetic	aesthetic	ADJ	JJ	Degree=Pos	2	amod	_	_
2	Appreciation	appreciation	NOUN	NN	Number=Sing	0	root	_	_
3	and	and	CCONJ	CC	_	5	cc	_	_
4	Spanish	Spanish	ADJ	JJ	Degree=Pos	5	amod	_	_
5	Art	art	NOUN	NN	Number=Sing	2	conj	_	SpaceAfter=No
6	:	:	PUNCT	:	_	2	punct	_	_

# sent_id = GUM_academic_art-2
# text = Insights from Eye-Tracking
# s_type=frag
1	Insights	insight	NOUN	NNS	Number=Plur	0	root	_	_
2	from	from	ADP	IN	_	3	case	_	_
3	Eye-Tracking	eye-tracking	NOUN	NN	Number=Sing	1	nmod	_	_
    """

    test_edu = """<segment id="1" parent="71" relname="span">Aesthetic Appreciation and Spanish Art :</segment>
    <segment id="2" parent="1" relname="elaboration">Insights from Eye-Tracking</segment>
    """

    rstp = rstParser()
    rstp.test_dependencies()
    pred = rstp.parse({"edu": test_edu, "xml": test_xml, "dep": test_conll, "filename": test_filename})
    print()
    print(pred["rst"])




if __name__ == "__main__":
    test_main()
