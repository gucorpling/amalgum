EDU segmentation was performed manually: in order to get RST output, you'll first need to use 
[ToNy](https://gitlab.inria.fr/andiamo/tony/) (or some other RST EDU segmenter) to segment your text.

Here is a rough outline of how to do it with ToNy:

1. Move `lib/tony_scripts/predict_amalgum.sh` and `lib/tony_scripts/train_gum6.sh` in this repo to 
`code/contextual_embeddings` in the ToNy repo.
2. Train a ToNy model using `train_gum6.sh`--consult the script for details. Training data needs to be in 
conllu format, where the sentences have been split and the beginning of an EDU is marked with `BeginSeg=Yes` in the
`misc` column of the first CONLLU token of that segment. The `rs32conll.py` script may be helpful for this.
3. Use `predict_amalgum.sh` to use the model you just trained to make predictions. The output will be in the same
conllu format--the script `conll2rs3.py` can help convert this into the RS3 format.
4. Put the predicted data under the `rst` folder in the output directory of the last NLP pipeline module that ran.