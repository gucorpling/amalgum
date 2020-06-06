from glob import glob
from conllu import parse
import xml.etree.ElementTree as ET


def format_tokens(tokens, base_index, begin_indices):
    for i, t in enumerate(tokens):
        line = ['_']*10
        line[0] = str(i + 1)
        line[1] = t
        if base_index + i in begin_indices:
            line[9] = "BeginSeg=Yes"
        yield "\t".join(line)


def process_file(filename):
    # load conllu gold
    with open(f'gold_dep/{filename[4:-4]}.conllu', 'r') as f:
        full_conllu = parse(f.read())

    # load rs3
    tree = ET.parse(filename)
    root = tree.getroot()

    # check identical toks
    rst_toks = [tok for seg in root.iter('segment') for tok in seg.text.split(' ')]
    conllu_toks = [tok for tl in full_conllu for tok in tl]
    assert len(rst_toks) == len(conllu_toks)
    for i in range(len(rst_toks)):
        assert conllu_toks[i]['form'] == rst_toks[i], f"{filename}:{i}: {conllu_toks[i]['form']} vs. {rst_toks[i]}"

    # get indexes at which a new segment began
    begin_segs = []
    i = 0
    for seg in root.iter('segment'):
        begin_segs.append(i)
        i += len(seg.text.split(' '))

    # make new bare conllu with sentence splits from gold conllu and BeginSeg annotations from rs3
    conllu = f"# newdoc id = {filename[4:-4]}\n"
    i = 0
    for sentence in full_conllu:
        conllu += "\n".join(format_tokens([t['form'] for t in sentence], i, begin_segs)) + "\n\n"
        i += len(sentence)
    return conllu


filenames = glob('rs3/*.rs3')
for filename in filenames:
    conllu = process_file(filename)
    with open(f'conllu/{filename[4:-4]}.conllu', 'w') as f:
        f.write(conllu)
