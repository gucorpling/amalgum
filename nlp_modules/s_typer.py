"""
s_typer.py

Takes conllu sentence parses and TTSGML and returns the same sentences with s_type comments in .conllu
and as XML <s> tag attributes in .xml
"""

from nlp_modules.base import NLPModule, PipelineDep
from lib.stype_classifier import STypeClassifier

class STyper(NLPModule):
    requires = (PipelineDep.S_SPLIT, PipelineDep.POS_TAG)
    provides = (PipelineDep.S_TYPE,)

    def __init__(self, opts):
        self.test_dependencies()
        self.styper = STypeClassifier()

    def test_dependencies(self):
        import numpy
        import scipy
        import sklearn
        import xgboost
        import pandas
        import joblib

    def resolve_stypes(self, doc_dict):
        preds = self.styper.predict(doc_dict["dep"])

        out_conll = []
        out_xml = []
        sents = doc_dict["dep"].strip().split("\n\n")
        for i, sent in enumerate(sents):
            s_type = "# s_type = " + preds[i]
            sent = s_type + "\n" + sent
            out_conll.append(sent)
        out_conll = "\n\n".join(out_conll) + "\n\n"

        counter = 0
        for line in doc_dict["xml"].split("\n"):
            if line == "<s>":
                line = '<s type="' + preds[counter] + '">'
                counter += 1
            out_xml.append(line)
        out_xml = "\n".join(out_xml)

        return {"xml": out_xml, "dep": out_conll}

    def run(self, input_dir, output_dir):


        # Identify a function that takes data and returns output at the document level
        processing_function = self.resolve_stypes

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function
        )

def test_main():
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

    test_conll = """1	Aesthetic	_	ADJ	JJ	_	2	amod	_	_
2	Appreciation	_	NOUN	NN	_	0	root	_	_
3	and	_	CCONJ	CC	_	5	cc	_	_
4	Spanish	_	ADJ	JJ	_	5	amod	_	_
5	Art	_	NOUN	NN	_	2	conj	_	_
6	:	_	PUNCT	:	_	2	punct	_	_

1	Insights	_	NOUN	NNS	_	0	root	_	_
2	from	_	ADP	IN	_	3	case	_	_
3	Eye-Tracking	_	NOUN	NN	_	1	nmod	_	_
"""

    stp = STyper()
    stp.test_dependencies()
    res = stp.resolve_stypes({"xml":test_xml,"dep":test_conll})
    print(res["xml"])
    print(res["dep"])

if __name__ == "__main__":
    test_main()

