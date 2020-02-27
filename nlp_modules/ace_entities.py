import os, io, re, sys, requests

from glob import glob
from nlp_modules.base import NLPModule, PipelineDep
from lib.shibuya_entities.ShibuyaEntities import ShibuyaEntities


class AceEntities(NLPModule):
    requires = (PipelineDep.PARSE)
    provides = (PipelineDep.ACE_ENTITIES)

    def __init__(self, serialnumber="200226_153935"):
        self.acedir = os.path.join('.', 'lib', 'shibuya_entities', 'data', 'amalgum')
        self.serialnumber = serialnumber
        self.shibuya = ShibuyaEntities()
        self.test_dependencies()

    def download_file(self, url, local_path):
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
        import torch
        import pandas, numpy
        self.download_file('https://corpling.uis.georgetown.edu/amir/download/sample_model_200226_153935.pt',
                           os.path.join('.','lib', 'shibuya_entities', 'dumps', 'sample_model_200226_153935.pt'))


    def parse(self, doc_dict):
        
        inputstr = doc_dict["dep"]
        assert '\n\n' in inputstr and '\t' in inputstr
        acegoldstr = self.shibuya.conllu2acegold(inputstr)
        
        with io.open(self.acedir + os.sep + 'amalgum.test', 'w', encoding='utf8') as ftest:
            ftest.write(acegoldstr)
        print("Step 1: File written to ACE format")

        # Pickle test data
        self.shibuya.gen_data(dataset="amalgum")
        print("Step 2: File converted to pickle")

        # predicts and outputs subtoks
        outputstr, _ = self.shibuya.predict(dataset="amalgum", serialnumber=self.serialnumber)
        print("Step 3: File predicted into BERT subtokens")

        # convert subtoks to toks
        outputstr = self.shibuya.subtok2tok(outputstr, acegoldstr)
        print("Step 4: File converted into tokens")

        return {"ace": outputstr}


    def run(self, input_dir, output_dir):
 

        # Identify a function that takes data and returns output at the document level
        processing_function = self.parse

        # use process_files, inherited from NLPModule, to apply this function to all docs
        # self.process_files(
        #     input_dir, output_dir, processing_function, multithreaded=False
        # )
        
        self.process_files_multiformat(input_dir, processing_function, output_dir, multithreaded=False)



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


    e = AceEntities()
    res = e.parse({"dep":test_conll,"filename":"autogum_voyage_doc001"})
    print(res["ace"])

if __name__ == "__main__":
    test_main()

