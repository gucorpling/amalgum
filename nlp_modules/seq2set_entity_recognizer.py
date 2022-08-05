"""
seq2set_entity_recognizer.py

Resolves entity types from conllu parses.
"""
from nlp_modules.base import NLPModule, PipelineDep
from ssn.args import eval_argparser
from ssn import input_reader
from ssn.config_reader import process_configs
from seq2set_evaluator import SSNTrainer


class Seq2setNNER(NLPModule):
    """
    Input: Conllu
    Output: Conllu
    """
    requires = (PipelineDep.PARSE,)
    provides = (PipelineDep.PARSE,)

    def __init__(self):
        self.test_dependencies()

    def test_dependencies(self):
        import torch
        import transformers
        import numpy
        import spacy
        import jinja2

    def resolve_entities(self, doc_dict):
        ssn_input = self._convert_input_format(doc_dict['dep'])
        json_out = self._predict(ssn_input)
        conllu_output = self._convert_output_format(doc_dict['dep'], json_out)
        return {"dep": conllu_output}

    def run(self, input_dir, output_dir):

        # Identify a function that takes data and returns output at the document level
        processing_function = self.resolve_entities

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(
            input_dir, output_dir, processing_function
        )

    def _convert_input_format(self, dep_doc):
        sents = dep_doc.strip().split('\n\n')
        input_data = []
        for sent in sents:
            json_sent = {'tokens': [], 'pos': [], 'ltokens': [], 'rtokens': [], 'entities': []}
            for line in sent.strip().split('\n'):
                if line.startswith('#'):
                    continue
                fields = line.split('\t')
                if '.' in fields[0] or '-' in fields[0]:
                    continue
                json_sent['tokens'].append(fields[1])
                json_sent['pos'].append(fields[4])
            input_data.append(json_sent)

        for sent_id, json_sent in enumerate(input_data):
            if sent_id > 0:
                input_data[sent_id]['ltokens'] = input_data[sent_id-1]['tokens']
            if sent_id < len(input_data) - 1:
                input_data[sent_id]['rtokens'] = input_data[sent_id+1]['tokens']
        return input_data

    def _convert_output_format(self, dep_doc, pred_data):
        gold_sents = dep_doc.strip().split('\n\n')

        pred_doc = []
        ent_id = 1
        pred_sent_id = 0
        for gold_sent_id, sent in enumerate(gold_sents):
            if sent.startswith('# newdoc id'):
                ent_id = 1
            gold_sent = sent.split('\n')
            pred_sent = pred_data[pred_sent_id]

            heading = []
            lines = []
            for line in gold_sent:
                if line.startswith('#'):
                    heading.append(line)
                else:
                    line_fields = line.split()
                    lines.append(line_fields)

            pred_ents = []
            for e in pred_sent['entities']:
                ent = [e['start'], e['end'] - 1, e['type']]
                if ent not in pred_ents:
                    pred_ents.append(ent)

            coref_fields = [''] * len(lines)
            for start, end, type in pred_ents:
                if start == end:
                    coref_fields[start] += f'({type}-{ent_id})'
                else:
                    coref_fields[start] += f'({type}-{ent_id}'
                    coref_fields[end] += f'{ent_id})'
                ent_id += 1

            for i in range(len(lines)):
                if coref_fields[i]:
                    if lines[i][-1] != '_':
                        lines[i][-1] += '|Entity=' + coref_fields[i]
                    else:
                        lines[i][-1] = 'Entity=' + coref_fields[i]

            new_sent = '\n'.join(heading) + '\n' + '\n'.join(['\t'.join(l) for l in lines])
            pred_doc.append(new_sent.strip())
            pred_sent_id += 1
        return '\n\n'.join(pred_doc)

    def _predict(self, input_data):
        run_args = process_configs(eval_argparser())
        trainer = SSNTrainer(run_args)
        return trainer.eval(input_data=input_data, types_path=run_args.types_path,
                            input_reader_cls=input_reader.JsonInputReader)


def test_main():
    test_conll = """# newdoc id = GUM_academic_art
# global.Entity = etype-eid-infstat-minspan-link-identity
# meta::dateCollected = 2017-09-13
# meta::dateCreated = 2017-08-08
# meta::dateModified = 2017-09-13
# meta::sourceURL = https://dh2017.adho.org/abstracts/333/333.pdf
# meta::speakerCount = 0
# meta::title = Aesthetic Appreciation and Spanish Art: Insights from Eye-Tracking
# sent_id = GUM_academic_art-1
# s_type = frag
# text = Aesthetic Appreciation and Spanish Art:
# newpar = head (2 s) | hi rend:::"bold blue" (2 s)
1	Aesthetic	aesthetic	ADJ	JJ	Degree=Pos	2	amod	2:amod	Discourse=organization-heading:1->57:8
2	Appreciation	appreciation	NOUN	NN	Number=Sing	0	root	0:root	_
3	and	and	CCONJ	CC	_	5	cc	5:cc	_
4	Spanish	Spanish	ADJ	JJ	Degree=Pos	5	amod	5:amod	_
5	Art	art	NOUN	NN	Number=Sing	2	conj	2:conj:and	SpaceAfter=No
6	:	:	PUNCT	:	_	2	punct	2:punct	_

# sent_id = GUM_academic_art-2
# s_type = frag
# text = Insights from Eye-Tracking
1	Insights	insight	NOUN	NNS	Number=Plur	0	root	0:root	Discourse=elaboration-additional:2->1:0
2	from	from	ADP	IN	_	5	case	5:case	_
3	Eye	eye	NOUN	NN	Number=Sing	5	compound	5:compound	SpaceAfter=No|XML=<w>
4	-	-	PUNCT	HYPH	_	3	punct	3:punct	SpaceAfter=No
5	Tracking	tracking	NOUN	NN	Number=Sing	1	nmod	1:nmod:from	XML=</w>

# sent_id = GUM_academic_art-3
# s_type = frag
# text = Claire Bailey-Ross claire.bailey-ross@port.ac.uk University of Portsmouth, United Kingdom
# newpar = p (4 s)
1	Claire	Claire	PROPN	NNP	Number=Sing	0	root	0:root	Discourse=attribution-positive:3->57:7
2	Bailey	Bailey	PROPN	NNP	Number=Sing	1	flat	1:flat	SpaceAfter=No|XML=<w>
3	-	-	PUNCT	HYPH	_	4	punct	4:punct	SpaceAfter=No
4	Ross	Ross	PROPN	NNP	Number=Sing	2	flat	2:flat	XML=</w>
5	claire.bailey-ross@port.ac.uk	claire.bailey-ross@port.ac.uk	PROPN	NNP	Number=Sing	1	list	1:list	_
6	University	University	PROPN	NNP	Number=Sing	1	list	1:list	_
7	of	of	ADP	IN	_	8	case	8:case	_
8	Portsmouth	Portsmouth	PROPN	NNP	Number=Sing	6	nmod	6:nmod:of	SpaceAfter=No
9	,	,	PUNCT	,	_	11	punct	11:punct	_
10	United	Unite	VERB	NNP	Tense=Past|VerbForm=Part	11	amod	11:amod	_
11	Kingdom	Kingdom	PROPN	NNP	Number=Sing	1	list	1:list	_

# sent_id = GUM_academic_art-4
# s_type = frag
# text = Andrew Beresford a.m.beresford@durham.ac.uk Durham University, United Kingdom
1	Andrew	Andrew	PROPN	NNP	Number=Sing	0	root	0:root	Discourse=joint-list_m:4->3:0
2	Beresford	Beresford	PROPN	NNP	Number=Sing	1	flat	1:flat	_
3	a.m.beresford@durham.ac.uk	a.m.beresford@durham.ac.uk	PROPN	NNP	Number=Sing	1	list	1:list	_
4	Durham	Durham	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	University	University	PROPN	NNP	Number=Sing	1	list	1:list	SpaceAfter=No
6	,	,	PUNCT	,	_	8	punct	8:punct	_
7	United	Unite	VERB	NNP	Tense=Past|VerbForm=Part	8	amod	8:amod	_
8	Kingdom	Kingdom	PROPN	NNP	Number=Sing	1	list	1:list	_

# sent_id = GUM_academic_art-5
# s_type = frag
# text = Daniel Smith daniel.smith2@durham.ac.uk Durham University, United Kingdom
1	Daniel	Daniel	PROPN	NNP	Number=Sing	0	root	0:root	Discourse=joint-list_m:5->3:0
2	Smith	Smith	PROPN	NNP	Number=Sing	1	flat	1:flat	_
3	daniel.smith2@durham.ac.uk	daniel.smith2@durham.ac.uk	PROPN	NNP	Number=Sing	1	list	1:list	_
4	Durham	Durham	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	University	University	PROPN	NNP	Number=Sing	1	list	1:list	SpaceAfter=No
6	,	,	PUNCT	,	_	8	punct	8:punct	_
7	United	Unite	VERB	NNP	Tense=Past|VerbForm=Part	8	amod	8:amod	_
8	Kingdom	Kingdom	PROPN	NNP	Number=Sing	1	list	1:list	_

# sent_id = GUM_academic_art-6
# s_type = frag
# text = Claire Warwick c.l.h.warwick@durham.ac.uk Durham University, United Kingdom
1	Claire	Claire	PROPN	NNP	Number=Sing	0	root	0:root	Discourse=joint-list_m:6->3:0
2	Warwick	Warwick	PROPN	NNP	Number=Sing	1	flat	1:flat	_
3	c.l.h.warwick@durham.ac.uk	c.l.h.warwick@durham.ac.uk	PROPN	NNP	Number=Sing	1	list	1:list	_
4	Durham	Durham	PROPN	NNP	Number=Sing	5	compound	5:compound	_
5	University	University	PROPN	NNP	Number=Sing	1	list	1:list	SpaceAfter=No
6	,	,	PUNCT	,	_	8	punct	8:punct	_
7	United	Unite	VERB	NNP	Tense=Past|VerbForm=Part	8	amod	8:amod	_
8	Kingdom	Kingdom	PROPN	NNP	Number=Sing	1	list	1:list	_

"""

    nner_model = Seq2setNNER()
    res = nner_model.resolve_entities({"dep": test_conll, "filename": "autogum_voyage_doc001"})
    print(res["dep"])


if __name__ == "__main__":
    test_main()
