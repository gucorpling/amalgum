"""
flair_pos_tagger.py

This module trains flair sequence labelers to predict POS and deprel for OTHER modules.
It is not the final amalgum POS tagger - it produces modules used by other modules,
such as the ensemble tagger in pos_tagger.py
"""


from argparse import ArgumentParser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, BertEmbeddings, OneHotEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
import os, sys, io
from .configuration import GUM_ROOT, EWT_ROOT
from glob import glob

if not GUM_ROOT.endswith(os.sep):
    GUM_ROOT += os.sep
if not EWT_ROOT.endswith(os.sep):
    EWT_ROOT += os.sep

ud_dev = ["GUM_interview_cyclone", "GUM_interview_gaming",
          "GUM_news_iodine", "GUM_news_homeopathic",
          "GUM_voyage_athens", "GUM_voyage_coron",
          "GUM_whow_joke", "GUM_whow_overalls",
          "GUM_bio_byron", "GUM_bio_emperor",
          "GUM_fiction_lunre", "GUM_fiction_beast",
          "GUM_academic_exposure", "GUM_academic_librarians",
          # "GUM_reddit_macroeconomics", "GUM_reddit_pandas",
          "GUM_speech_impeachment", "GUM_textbook_cognition",
          "GUM_vlog_radiology", "GUM_conversation_grounded"]
ud_test = ["GUM_interview_libertarian", "GUM_interview_hill",
           "GUM_news_nasa", "GUM_news_sensitive",
           "GUM_voyage_oakland", "GUM_voyage_vavau",
           "GUM_whow_mice", "GUM_whow_cactus",
           "GUM_fiction_falling", "GUM_fiction_teeth",
           "GUM_bio_jespersen", "GUM_bio_dvorak",
           "GUM_academic_eegimaa", "GUM_academic_discrimination",
           # "GUM_reddit_escape", "GUM_reddit_monsters",
           "GUM_speech_austria", "GUM_textbook_chemistry",
           "GUM_vlog_studying", "GUM_conversation_retirement"]


def make_gum_pos(tag="xpos", tag2=None, corpus="gum"):
    if corpus == "gum":
        gum_target = GUM_ROOT + os.sep.join(['_build','target','dep','not-to-release']) + os.sep
    else:
        gum_target = EWT_ROOT
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
    else:
        colnum2 = 4
    for file_ in files:
        output = []
        lines = io.open(file_,encoding="utf8").readlines()
        docname = os.path.basename(file_).replace(".conllu","")
        for line in lines:
            if "\t" in line:
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    continue
                if tag2 is not None:
                    output.append(fields[1] + "\t" + fields[colnum] + "\t" + fields[colnum2])
                else:
                    output.append(fields[1] + "\t" + fields[colnum])
            elif len(line.strip()) == 0:
                if output[-1] != "":
                    output.append("")
        if docname in ud_dev or "-dev" in docname:
            dev += "\n".join(output)
        elif docname in ud_test or "-test" in docname:
            test += "\n".join(output)
        else:
            train += "\n".join(output)
    with io.open("pos-dependencies" + os.sep + corpus + "_"+tag+"_train.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write(train)
    with io.open("pos-dependencies" + os.sep + corpus + "_"+tag+"_dev.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write(dev)
    with io.open("pos-dependencies" + os.sep + corpus + "_"+tag+"_test.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write(test)

def train(tag="xpos",tag2=None,corpus="gum"):
    # Prevent CUDA Launch Failure random error, but slower:
    import torch
    torch.backends.cudnn.enabled = False
    # Or:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 1. get the corpus
    # this is the folder in which train, test and dev files reside
    data_folder = "pos-dependencies"  + os.sep

    # init a corpus using column format, data folder and the names of the train, dev and test files

    # define columns
    columns = {0: "text", 1: "pos"}
    if tag2 is not None:
        columns[2] = tag2

    make_gum_pos(tag=tag,tag2=tag2,corpus=corpus)

    corpus: Corpus = ColumnCorpus(
        data_folder, columns,
        train_file=corpus+"_"+tag+"_train.txt",
        test_file=corpus+"_"+tag+"_test.txt",
        dev_file=corpus+"_"+tag+"_dev.txt",
    )

    # 2. what tag do we want to predict?
    tag_type = 'pos'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    # 4. initialize embeddings
    embedding_types = [

        #WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        #FlairEmbeddings('news-forward'),
        #FlairEmbeddings('news-backward'),
        #BertEmbeddings('distilbert-base-cased')
        TransformerWordEmbeddings('google/electra-base-discriminator')
    ]


    if tag2 is not None:
        pass
        c_emb = OneHotEmbeddings(corpus=corpus,field=tag2,embedding_length=8)
        embedding_types.append(c_emb)

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings, #electra,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True,
                                            use_rnn=True)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train("pos-dependencies" + os.sep + 'flair_tagger',
                  learning_rate=0.1,
                  mini_batch_size=15,
                  max_epochs=150)

def predict(corpus="gum",tag="xpos",tag2=None,in_format="flair",out_format="conllu",in_path=None):
    tag_string = tag if tag2 is None else tag + "_" + tag2
    model_name = "pos-dependencies" + os.sep +"flair_tagger/best-model_"+tag_string+"_"+corpus+".pt"
    if corpus=="ontonotes":
        model_name = "pos"
    elif corpus=="ner":
        model_name="ner"
    model = SequenceTagger.load(model_name)

    if tag =="upos":
        tagcol = 3
    elif tag=="deprel":
        tagcol = 7
    else:
        tagcol = 4
    if tag2 =="upos":
        tagcol2 = 3
    elif tag2=="deprel":
        tagcol2 = 7
    elif tag2=="xpos":
        tagcol2 = 4

    if in_path is None:
        in_path = "pos-dependencies" + os.sep + "GUM_pos_dev.txt"
    dev = io.open(in_path,encoding="utf8").read()
    sents = []
    words = []
    annos = []
    true_tags = []
    true_pos = []
    for line in dev.split("\n"):
        if len(line.strip())==0:
            if len(words) > 0:
                sents.append(Sentence(" ".join(words),use_tokenizer=lambda x:x.split(" ")))
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
                    if tag2 is None:
                        words.append(line.split("\t")[0])
                        true_tags.append(line.split("\t")[1])
                    else:
                        words.append(line.split("\t")[0])
                        annos.append(line.split("\t")[1])
                        true_tags.append(line.split("\t")[2])
                else:
                    if tag2 is None:
                        words.append(line.split("\t")[1])
                    else:
                        words.append(line.split("\t")[1])
                        annos.append(line.split("\t")[tagcol2])
                    true_tags.append(line.split("\t")[tagcol])
                    true_pos.append(line.split("\t")[4])

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

    do_postprocess = True
    if tag != "xpos":
        do_postprocess= False
    if do_postprocess:
        preds, scores = post_process(words, preds, scores)

    toknum = 0
    output = []
    for i, sent in enumerate(sents):
        tid=1
        if i>0 and out_format=="conllu":
            output.append("")
        for tok in sent.tokens:
            pred = preds[toknum]
            score = str(scores[toknum])
            if len(score)>5:
                score = score[:5]
            if out_format == "conllu":
                if tag2 is None:
                    fields = [str(tid),tok.text,"_",pred,"_","_","_","_","_"]
                else:
                    fields = [str(tid), tok.text, "_", pred, tok.labels[0].value, "_", "_", "_", "_"]
                output.append("\t".join(fields))
                tid+=1
            elif out_format == "xg":
                output.append("\t".join([pred, tok.text, score]))
            else:
                true_tag = true_tags[toknum]
                corr = "T" if true_tag == pred else "F"
                output.append("\t".join([pred, true_tag, corr, score, tok.text, true_pos[toknum]]))
            toknum += 1

    ext = tag + ".conllu" if out_format == "conllu" else "txt"
    partition = "test" if "test" in in_path else "dev"
    with io.open("pos-dependencies" +os.sep + "flair-"+corpus+"-" + tag + "-"+partition+"-pred." + ext,'w',encoding="utf8",newline="\n") as f:
        f.write("\n".join(output))


def post_process(word_list, pred_list, score_list, softmax_list=None):
    """
    Implement a subset of closed-class words that can only take one of their attested closed class POS tags
    """
    output = []

    closed = {"except":["IN"],
              "or":["CC"],
              "another":["DT"],
              "be":["VB"]
              }
    # case marking VVG can never be IN:
    vbg_preps = {("including","IN"):"VBG",("according","IN"):"VBG",("depending","IN"):"VBG",("following","IN"):"VBG",("involving","IN"):"VBG",
                 ("regarding","IN"):"VBG",("concerning","IN"):"VBG"}

    top100 = {",":",",".":".","of":"IN","is":"VBZ","you":"PRP","for":"IN","was":"VBD","with":"IN","The":"DT","are":"VBP",")":"-RRB-","(":"-LRB-","at":"IN","this":"DT","from":"IN","or":"CC","not":"RB","his":"PRP$","they":"PRP","an":"DT","we":"PRP","n't":"RB","he":"PRP","[":"-LRB-","]":"-RRB-","has":"VBZ","my":"PRP$","their":"PRP$","It":"PRP","were":"VBD","In":"IN","if":"IN","would":"MD","”":"''",";":":","into":"IN","when":"WRB","You":"PRP","also":"RB","she":"PRP","our":"PRP$","been":"VBN","who":"WP","We":"PRP","time":"NN","He":"PRP","This":"DT","its":"PRP$","did":"VBD","two":"CD","these":"DT","many":"JJ","And":"CC","!":".","should":"MD","because":"IN","how":"WRB","If":"IN","n’t":"RB","'re":"VBP","him":"PRP","'m":"VBP","city":"NN","could":"MD","may":"MD","years":"NNS","She":"PRP","really":"RB","now":"RB","new":"JJ","something":"NN","here":"RB","world":"NN","They":"PRP","life":"NN","But":"CC","year":"NN","us":"PRP","between":"IN","different":"JJ","those":"DT","language":"NN","does":"VBZ","same":"JJ","going":"VBG","United":"NNP","day":"NN","few":"JJ","For":"IN","every":"DT","important":"JJ","When":"WRB","things":"NNS","during":"IN","might":"MD","kind":"NN","How":"WRB","system":"NN","thing":"NN","example":"NN","another":"DT","small":"JJ","until":"IN","information":"NN","away":"RB"}

    scores = []

    #VBG must end in ing/in; VBN may not
    for i, word in enumerate(word_list):
        pred = pred_list[i]
        score = score_list[i]
        if word in top100:
            output.append(top100[word])
            scores.append("_")
        elif (word.lower(),pred) in vbg_preps:
            output.append(vbg_preps[(word.lower(),pred)])
            scores.append("_")
        else:
            output.append(pred)
            scores.append(score)

    # Also VB+RP/RB disambig from large list? PTB+ON?

    return output, scores


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-m","--mode",choices=["train","predict"],default="predict")
    p.add_argument("-f","--file",default=None,help="Blank for training, blank predict for eval, or file to run predict on")
    p.add_argument("-i","--input_format",choices=["flair","conllu"],default="flair",help="flair two column training format or conllu")
    p.add_argument("-o","--output_format",choices=["flair","conllu","xg"],default="conllu",help="flair two column training format or conllu")
    p.add_argument("-t","--tag",choices=["xpos","upos","deprel"],default="xpos",help="tag to learn/predict")
    p.add_argument("-t2","--tag2",choices=["xpos","upos","deprel"],default=None,help="auxiliary tag with features")
    p.add_argument("-c","--corpus",default="gum",help="corpus name for model file name")

    opts = p.parse_args()

    if opts.mode == "train":
        train(tag=opts.tag,tag2=opts.tag2,corpus=opts.corpus)
    else:
        predict(corpus=opts.corpus, tag=opts.tag, tag2=opts.tag2,
                in_format=opts.input_format, out_format=opts.output_format,
                in_path=opts.file)

