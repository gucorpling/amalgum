# Development
## Dependencies
1. Set up a conda environment:

```sh
# OPTION 1 (recommended): using env.yml
conda env create -f env.yml

# OPTION 2: using requirements.txt
conda create --name amalgum python=3.7
conda activate amalgum
pip install -r requirements.txt
```

2. *(optional)* If you have CUDA-capable hardware, add CUDA support: `conda install "pytorch<1.6" torchvision cudatoolkit -c pytorch`

3. Download `punkt`: `python -c "import nltk; nltk.download('punkt')"`

4. *(For Windows platforms)*, Download and install the 64-bit JRE and setup the JAVA_HOME environment variable with the location of this JRE (typically in C:\Program Files\Java\<your jre folder>)

## NLP Pipeline
Invoke nlp_controller.py on the tiny subset to ensure the pipeline is working properly:

```bash
python nlp_controller.py target -i out_tiny
```
## Adding an NLPModule

* Make a new file in [`nlp_modules`](https://github.com/gucorpling/amalgum/tree/master/nlp_modules)

- Make a subclass of [`NLPModule`](https://github.com/gucorpling/amalgum/blob/master/nlp_modules/base.py#L29L250).
  - You will need to implement the methods
    - `__init__`, the constructor
    - `test_dependencies`, which should be used to download any static files (e.g. data, serialized models) that are required for your module's operation
    - `run`, which is the method that the controller will use to invoke your module.
  - **In addition**, you will also need to use the class attributes `requires` and `provides` to declare what kinds of NLP processing your module will expect, and what kind of processing it will provide, respectively, expressed using [values of the `PipelineDep` enum](https://github.com/gucorpling/amalgum/blob/master/nlp_modules/base.py#L12L23). (For instance, [for a POS tagger, `requires = (PipelineDep.TOKENIZE,)`, and `provides = PipelineDep.POS_TAG`](https://github.com/gucorpling/amalgum/blob/master/nlp_modules/tt_tagger.py#L10).)
  - The remaining methods, `process_files` and `process_files_multiformat`, are convenience functions that you should consider using in your implementation of `run`.
  - See the [TreeTagger POS tagging module](https://github.com/gucorpling/amalgum/blob/master/nlp_modules/tt_tagger.py#L10) for a small example.

- Register your module [in `nlp_controller.py`](https://github.com/gucorpling/amalgum/blob/master/nlp_controller.py#L29)

- Depending on what's appropriate, either add your module to [`nlp_controller.py`'s `--modules` flag's default value](https://github.com/gucorpling/amalgum/blob/master/nlp_controller.py#L152L159), or invoke `nlp_controller.py` with your module included in `--modules`.
