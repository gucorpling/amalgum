
from glob import glob
from conllu import parse
from lxml import etree as ET

template = """<rst><header><relations><rel name="antithesis" type="rst"/><rel name="attribution" type="rst"/><rel name="background" type="rst"/><rel name="cause" type="rst"/><rel name="circumstance" type="rst"/><rel name="concession" type="rst"/><rel name="condition" type="rst"/><rel name="contrast" type="multinuc"/><rel name="elaboration" type="rst"/><rel name="evaluation" type="rst"/><rel name="evidence" type="rst"/><rel name="joint" type="multinuc"/><rel name="justify" type="rst"/><rel name="manner" type="rst"/><rel name="means" type="rst"/><rel name="motivation" type="rst"/><rel name="preparation" type="rst"/><rel name="purpose" type="rst"/><rel name="question" type="rst"/><rel name="restatement" type="multinuc"/><rel name="restatement" type="rst"/><rel name="result" type="rst"/><rel name="same-unit" type="multinuc"/><rel name="sequence" type="multinuc"/><rel name="solutionhood" type="rst"/></relations></header><body></body></rst>"""


with open('gum6_test.predictions.conllu', 'r') as f:
    preds = parse(f.read())
filenames = sorted(glob('amalgum_conllu/*.conllu'))
#filenames = sorted(glob('conllu/dev/*.conllu'))

i = 0
for filename in filenames:
    rs3 = ET.fromstring(template)
    body = rs3.find('body')
    rs3_filename = 'rs3/' + filename.split('/')[-1][:-6] + 'rs3'

    # ensure we're reading the right lines from the pred doc for this doc
    with open(filename, 'r') as f:
        gold_doc = parse(f.read())
    pred_doc = preds[i:i+len(gold_doc)]
    for j, gold_sent in enumerate(gold_doc):
        pred_sent = pred_doc[j]
        assert len(gold_sent) == len(pred_sent)
        for k in range(len(gold_sent)):
            assert gold_sent[k]['form'] == pred_sent[k]['form']

    # go through tokens and make a new segment every time we find BeginSeg=Yes on a pred token
    pred_toks = [tok for sent in pred_doc for tok in sent]
    current_segment = []
    segment_count = 0
    for tok in pred_toks:
        if tok['misc'] is not None and 'BeginSeg' in tok['misc'] and tok['misc']['BeginSeg'] == 'Yes':
            if len(current_segment) > 0:
                segment_count += 1
                seg = ET.SubElement(body, 'segment', id=str(segment_count))
                seg.text = ' '.join(current_segment)
                current_segment = []
        current_segment.append(tok['form'])
    if len(current_segment) > 0:
        segment_count += 1
        seg = ET.SubElement(body, 'segment', id=str(segment_count))
        seg.text = ' '.join(current_segment)

    tree = ET.ElementTree(rs3)
    tree.write(rs3_filename, encoding="UTF-8", pretty_print=True)
    i += len(gold_doc)


