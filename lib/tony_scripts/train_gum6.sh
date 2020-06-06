# usage
#  sh expes.sh dataset config model

export DATASET=gum6
export CONFIG=tok
export MODEL=bert

# english models
export BERT_VOCAB="bert-base-cased"
export BERT_WEIGHTS="bert-base-cased"                                                                                                   

export EVAL=dev

export GOLD_BASE="../../data/"
export CONV="data_converted/"
export TRAIN_DATA_PATH=${CONV}${DATASET}"_train.ner."${CONFIG}
export TEST_A_PATH=${CONV}${DATASET}"_"${EVAL}".ner."${CONFIG}
export OUTPUT=${DATASET}"_"${MODEL}
export GOLD=${GOLD_BASE}${DATASET}"/"${DATASET}"_"${EVAL}"."${CONFIG}

set -x
echo "converting to ner format -> in data_converted ..."
python conv2ner.py "../../data_ssplit/"${DATASET}"/"${DATASET}"_train."${CONFIG} > ${CONV}/${DATASET}"_train.ner."${CONFIG}
python conv2ner.py "../../data_ssplit/"${DATASET}"/"${DATASET}"_dev."${CONFIG} > ${CONV}/${DATASET}"_dev.ner."${CONFIG}

# train with config in ner_elmo ou ner_bert.jsonnet; the config references explicitely variables TRAIN_DATA_PATH and TEST_A_PATH
allennlp train -s Results_${CONFIG}/results_${OUTPUT} configs/bert.jsonnet
# predict with model -> outputs json
allennlp predict --use-dataset-reader --output-file Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.json Results_${CONFIG}/results_${OUTPUT}/model.tar.gz ${TEST_A_PATH}


# so, I need to:
# - put gum 6 conllu in ../../data_ssplit/amalgum_{train,dev,test}.split.tok
# - train on gum 6
# - put amalgum conllu in ../../data_ssplit/amalgum_{train,dev,test}.split.tok
# - conv2ner.py it into data_converted/

# - need a script that'll convert rs3 -> conllu, conllu -> rs3
