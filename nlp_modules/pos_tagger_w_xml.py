from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    StackedEmbeddings,
    OneHotEmbeddings,
    TransformerWordEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from glob import glob
import os
from argparse import ArgumentParser
import sys

GUM_ROOT = 'data/not-to-release/'

if not GUM_ROOT.endswith(os.sep):
    GUM_ROOT += os.sep

ud_dev = ["GUM_interview_cyclone", "GUM_interview_gaming",
          "GUM_news_iodine", "GUM_news_homeopathic",
          "GUM_voyage_athens", "GUM_voyage_coron",
          "GUM_whow_joke", "GUM_whow_overalls",
          "GUM_bio_byron", "GUM_bio_emperor",
          "GUM_fiction_lunre", "GUM_fiction_beast",
          "GUM_academic_exposure", "GUM_academic_librarians",
          "GUM_reddit_macroeconomics", "GUM_reddit_pandas",
          "GUM_speech_impeachment","GUM_speech_inauguration",
          "GUM_conversation_grounded","GUM_conversation_risk"
          "GUM_textbook_governments","GUM_textbook_labor",
          "GUM_vlog_portland","GUM_vlog_radiology"]

ud_test = ["GUM_interview_libertarian", "GUM_interview_hill",
           "GUM_news_nasa", "GUM_news_sensitive",
           "GUM_voyage_oakland", "GUM_voyage_vavau",
           "GUM_whow_mice", "GUM_whow_cactus",
           "GUM_fiction_falling", "GUM_fiction_teeth",
           "GUM_bio_jespersen", "GUM_bio_dvorak",
           "GUM_academic_eegimaa", "GUM_academic_discrimination",
           "GUM_reddit_escape", "GUM_reddit_monsters",
           "GUM_speech_austria", "GUM_speech_newzealand",
           "GUM_textbook_chemistry","GUM_textbook_union",
           "GUM_vlog_studying", "GUM_vlog_london",
           "GUM_conversation_retirement","GUM_conversation_lambada"]


def make_gum_pos(tag="upos", tag2="xml", corpus="gum"):
    gum_target = GUM_ROOT

    files = glob(gum_target + "*.conllu")
    train = test = dev = ""
    if tag == "upos":
        colnum = 3
    elif tag == "deprel":
        colnum = 7
    else:
        colnum = 4
    if tag2 == "upos":
        colnum2 = 3
    elif tag2 == "deprel":
        colnum2 = 7
    elif tag2 == "xml":
        colnum2 = 9
    else:
        colnum2 = 4
    for file_ in files:
        output = []
        lines = open(file_, encoding="utf8").readlines()
        docname = os.path.basename(file_).replace(".conllu", "")
        counter_head = 0
        is_begin = False
        inside = False
        for line in lines:
            if "\t" not in line:
                counter_head -= 1
            if "newpar" in line and "head" in line:
                sp = line.rstrip().split(' ')
                indd = sp.index("head")
                for kk in sp[indd:]:
                    if "(" in kk:
                        counter_head = int(kk[1:])
                        break
            if "\t" in line:
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    continue
                if "SpaceAfter=No" in fields[colnum2] and not inside:
                    is_begin = True
                    inside = True
                #if "SpaceAfter=No" not in fields[colnum2] and inside:
                #    inside = False
                if tag2 is not None:
                    if counter_head > 0 and int(fields[0]) == 1:
                        output.append(fields[1] + "\t" + fields[colnum] + "\t" + "B-head")
                    elif counter_head > 0 and int(fields[0]) > 1:
                        output.append(fields[1] + "\t" + fields[colnum] + "\t" + "I-head")
                    elif is_begin:
                        is_begin = False
                        output.append(fields[1] + "\t" + fields[colnum] + "\t" + "B-word")
                    elif inside:
                        output.append(fields[1] + "\t" + fields[colnum] + "\t" + "I-word")
                    else:
                        output.append(fields[1] + "\t" + fields[colnum] + "\t" + "O")
                else:
                    output.append(fields[1] + "\t" + fields[colnum])
                if "SpaceAfter=No" not in fields[colnum2] and inside:
                    inside = False
            elif len(line.strip()) == 0:
                if output[-1] != "":
                    output.append("")
        if docname in ud_dev or "-dev" in docname:
            dev += "\n".join(output)
        elif docname in ud_test or "-test" in docname:
            test += "\n".join(output)
        else:
            train += "\n".join(output)
    with open("data" + os.sep + "processed" + os.sep + corpus + "_" + tag + "_train.txt", 'w', encoding="utf8",
              newline="\n") as f:
        f.write(train)
    with open("data" + os.sep + "processed" + os.sep + corpus + "_" + tag + "_dev.txt", 'w', encoding="utf8",
              newline="\n") as f:
        f.write(dev)
    with open("data" + os.sep + "processed" + os.sep + corpus + "_" + tag + "_test.txt", 'w', encoding="utf8",
              newline="\n") as f:
        f.write(test)


def train(tag="xpos", tag2="xml", corpus="gum"):
    # init a corpus using column format, data folder and the names of the train, dev and test files

    # define columns
    columns = {0: "text", 1: "pos", 3: "xml"}
    data_folder = "data" + os.sep + "processed" + os.sep

    make_gum_pos(tag=tag, tag2=tag2, corpus=corpus)

    corpus: Corpus = ColumnCorpus(
        data_folder, columns,
        train_file=corpus + "_" + tag + "_train.txt",
        test_file=corpus + "_" + tag + "_test.txt",
        dev_file=corpus + "_" + tag + "_dev.txt",
    )

    # 2. what tag do we want to predict?
    tag_type = 'pos'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    # 4. initialize embeddings
    embeddings_1 = TransformerWordEmbeddings('bert-base-uncased')
    embeddings_2 = OneHotEmbeddings.from_corpus(corpus=corpus, field='xml', embedding_length=8)

    embedding_types = [
        embeddings_1,
        embeddings_2
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)
    #embeddings = embeddings_2

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True,
                            use_rnn=True)

    #tagger.save("trained_taggers/v1-march28")
    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('trained_taggers/v2/',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=100)


def predict(corpus="gum", tag="pos", tag2="xml", in_format="flair", out_format="info", in_path=None):
    model_name = "trained_taggers/v2/final-model.pt"
    model = SequenceTagger.load(model_name)

    if in_path is None:
        in_path = "data" + os.sep + "processed" + os.sep + "gum_xpos_dev.txt"
    dev = open(in_path, encoding="utf8").read()
    sents = []
    words = []
    annos = []
    true_tags = []
    true_pos = []
    for line in dev.split("\n"):
        if len(line.strip()) == 0:
            if len(words) > 0:
                sents.append(Sentence(" ".join(words), use_tokenizer=lambda x: x.split(" ")))
                if tag2 is not None:
                    for i, word in enumerate(sents[-1]):
                        word.add_tag(tag2, annos[i])
                    annos = []
                words = []
        else:
            if "\t" in line:
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    continue
                if in_format == "flair":
                    words.append(line.split("\t")[0])
                    annos.append(line.split("\t")[2])
                    true_tags.append(line.split("\t")[1])

    # predict tags and print
    model.predict(sents, all_tag_prob=True)

    preds = []
    scores = []
    words = []
    for i, sent in enumerate(sents):
        for tok in sent.tokens:
            if tag2 is None:
                pred = tok.labels[0].value
                score = str(tok.labels[0].score)
            else:
                pred = tok.labels[1].value
                score = str(tok.labels[1].score)
            preds.append(pred)
            scores.append(score)
            words.append(tok.text)

    toknum = 0
    output = []
    for i, sent in enumerate(sents):
        tid = 1
        if i > 0 and out_format == "conllu":
            output.append("")
        for tok in sent.tokens:
            pred = preds[toknum]
            score = str(scores[toknum])
            if len(score) > 5:
                score = score[:5]
            if out_format == "conllu":
                if tag2 is None:
                    fields = [str(tid), tok.text, "_", pred, "_", "_", "_", "_", "_"]
                else:
                    fields = [str(tid), tok.text, "_", pred, tok.labels[1].value, "_", "_", "_", "_"]
                output.append("\t".join(fields))
                tid += 1
            elif out_format == "xg":
                output.append("\t".join([pred, tok.text, score]))
            else:
                true_tag = true_tags[toknum]
                corr = "T" if true_tag == pred else "F"
                output.append("\t".join([pred, true_tag, corr, score, tok.text, true_pos[toknum]]))
            toknum += 1

    ext = tag + ".conllu" if out_format == "conllu" else "txt"
    partition = "test" if "test" in in_path else "dev"
    with open("out" + os.sep + "flair-" + corpus + "-" + tag + "-" + partition + "-pred." + ext, 'w',
              encoding="utf8", newline="\n") as f:
        f.write("\n".join(output))


if __name__ == "__main__":
    sys.setrecursionlimit(2500)
    p = ArgumentParser()
    p.add_argument("-m", "--mode", choices=["train", "predict"], default="train")
    p.add_argument("-f", "--file", default=None,
                   help="Blank for training, blank predict for eval, or file to run predict on")
    p.add_argument("-i", "--input_format", choices=["flair", "conllu"], default="flair",
                   help="flair two column training format or conllu")
    p.add_argument("-o", "--output_format", choices=["flair", "conllu", "xg"], default="conllu",
                   help="flair two column training format or conllu")
    p.add_argument("-t", "--tag", choices=["xpos", "upos", "deprel"], default="xpos", help="tag to learn/predict")
    p.add_argument("-t2", "--tag2", choices=["xpos", "upos", "deprel"], default="xml",
                   help="auxiliary tag with features")
    p.add_argument("-c", "--corpus", default="gum", help="corpus name for model file name")

    opts = p.parse_args()

    if opts.mode == "train":
        train(tag=opts.tag, tag2=opts.tag2, corpus=opts.corpus)
    else:
        predict(corpus=opts.corpus, tag=opts.tag, tag2=opts.tag2,
                in_format=opts.input_format, out_format=opts.output_format,
                in_path=opts.file)

