"""
xrenner_coreferencer.py

Resolves entity types and coreference from conllu parses.
Optionally consults external sequence labels in ACE format for entity types.
"""

from nlp_modules.base import NLPModule, PipelineDep
from xrenner import Xrenner

class XrennerCoref(NLPModule):
    # Note this module requires PARSE to contain s_type comments like:
    # s_type = decl
    requires = (PipelineDep.PARSE, )
    provides = (PipelineDep.TSV_OUT,)
    # To use ACE format entity type predictions from and external tool, comment out above and uncomment this:
    # provides = (PipelineDep.TSV_OUT, PipelineDep.ACE_ENTITIES)

    def __init__(self, rule_based=False, no_rnn=False, use_oracle=False):
        """

        :param rule_based: whether to turn off machine learning coref classification (much faster, less accurate)
        :param no_rnn: whether to turn off internal neural sequence labeler for classifying xrenner entity span
        :param use_oracle: whether to consult input from an external sequence labeler IFF it has matching spans
        """
        self.rule_base = rule_based
        self.no_rnn = no_rnn
        self.use_oracle = use_oracle
        self.test_dependencies()
        # Make an Xrenner object to check models
        self.xrenner = Xrenner(model="eng", override="GUM", rule_based=True, no_seq=True)

        # Make sure neural sequence labeler model is installed and download if not
        self.xrenner.check_model()

        # Actual Xrenner object with models
        self.xrenner = Xrenner(model="eng", override="GUM", rule_based=rule_based, no_seq=no_rnn)

    def test_dependencies(self):
        import six
        import depedit
        import xrenner
        if not self.rule_base:
            import xgboost
        if not self.no_rnn:
            import flair

    def resolve_entities_coref(self, doc_dict):
        self.xrenner.set_doc_name(doc_dict["filename"])
        if self.use_oracle:
            # Only used if ace format data is provided with entity type predictions from an external tool
            self.xrenner.lex.read_oracle(doc_dict["ace"], as_text=True)

        tsv_output = self.xrenner.analyze(doc_dict["dep"], "webannotsv")

        return {"tsv": tsv_output}

    def run(self, input_dir, output_dir):

        # Identify a function that takes data and returns output at the document level
        processing_function = self.resolve_entities_coref

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function
        )


def test_main():

    test_conll = """# newdoc id = GUM_voyage_thailand
# sent_id = GUM_voyage_thailand-1
# text = Respect
# s_type=frag
1	Respect	respect	NOUN	NN	Number=Sing	0	root	_	_

# sent_id = GUM_voyage_thailand-2
# text = Thais are a polite people and, while remarkably tolerant of foreigners gallivanting on their beaches and with their women, youll find that you will get more respect if you in turn treat them and their customs with respect.
# s_type=decl
1	Thais	Thai	PROPN	NNPS	Number=Plur	5	nsubj	_	_
2	are	be	AUX	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	5	cop	_	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	5	det	_	_
4	polite	polite	ADJ	JJ	Degree=Pos	5	amod	_	_
5	people	people	NOUN	NNS	Number=Plur	0	root	_	_
6	and	and	CCONJ	CC	_	24	cc	_	SpaceAfter=No
7	,	,	PUNCT	,	_	6	punct	_	_
8	while	while	SCONJ	IN	_	10	mark	_	_
9	remarkably	remarkably	ADV	RB	_	10	advmod	_	_
10	tolerant	tolerant	ADJ	JJ	Degree=Pos	24	advcl	_	_
11	of	of	ADP	IN	_	12	case	_	_
12	foreigners	foreigner	NOUN	NNS	Number=Plur	10	obl	_	_
13	gallivanting	gallivant	VERB	VBG	VerbForm=Ger	12	acl	_	_
14	on	on	ADP	IN	_	16	case	_	_
15	their	their	PRON	PRP$	Number=Plur|Person=3|Poss=Yes|PronType=Prs	16	nmod:poss	_	_
16	beaches	beach	NOUN	NNS	Number=Plur	13	obl	_	_
17	and	and	CCONJ	CC	_	13	cc	_	_
18	with	with	ADP	IN	_	20	case	_	_
19	their	their	PRON	PRP$	Number=Plur|Person=3|Poss=Yes|PronType=Prs	20	nmod:poss	_	_
20	women	woman	NOUN	NNS	Number=Plur	13	obl	_	SpaceAfter=No
21	,	,	PUNCT	,	_	10	punct	_	_
22	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	24	nsubj	_	SpaceAfter=No
23	ll	will	AUX	MD	VerbForm=Fin	24	aux	_	_
24	find	find	VERB	VB	VerbForm=Inf	5	conj	_	_
25	that	that	SCONJ	IN	_	28	mark	_	_
26	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	28	nsubj	_	_
27	will	will	AUX	MD	VerbForm=Fin	28	aux	_	_
28	get	get	VERB	VB	VerbForm=Inf	24	ccomp	_	_
29	more	more	ADJ	JJR	Degree=Cmp	30	amod	_	_
30	respect	respect	NOUN	NN	Number=Sing	28	obj	_	_
31	if	if	SCONJ	IN	_	35	mark	_	_
32	you	you	PRON	PRP	Case=Nom|Person=2|PronType=Prs	35	nsubj	_	_
33	in	in	ADP	IN	_	34	case	_	_
34	turn	turn	NOUN	NN	Number=Sing	35	obl	_	_
35	treat	treat	VERB	VBP	Mood=Ind|Tense=Pres|VerbForm=Fin	28	advcl	_	_
36	them	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	35	obj	_	_
37	and	and	CCONJ	CC	_	35	cc	_	_
38	their	their	PRON	PRP$	Number=Plur|Person=3|Poss=Yes|PronType=Prs	39	nmod:poss	_	_
39	customs	custom	NOUN	NNS	Number=Plur	35	obj	_	_
40	with	with	ADP	IN	_	41	case	_	_
41	respect	respect	NOUN	NN	Number=Sing	35	obl	_	SpaceAfter=No
42	.	.	PUNCT	.	_	5	punct	_	_

"""

    test_ace = """Respect
1,2 abstract
1,2 abstract

Thais are a polite people and , while remarkably tolerant of foreigners gallivanting on their beaches and with their women , you ll find that you will get more respect if you in turn treat them and their customs with respect .
1,2 person|3,6 person|12,13 person|15,17 place|19,21 person|22,23 person|26,27 person|29,31 abstract|32,33 person|36,37 person|38,40 object|41,42 abstract
1,2 place|3,6 person|12,13 person|15,16 person|15,17 place|19,20 person|19,21 person|26,27 person|29,31 abstract|32,33 person|36,37 person|38,40 abstract|41,42 abstract
"""

    xr = XrennerCoref()
    xr.test_dependencies()
    # res = xr.resolve_entities_coref({"ace":test_ace,"dep":test_conll,"filename":"autogum_voyage_doc001"})
    res = xr.resolve_entities_coref({"dep":test_conll,"filename":"autogum_voyage_doc001"})
    print(res["tsv"])


if __name__ == "__main__":
    test_main()

