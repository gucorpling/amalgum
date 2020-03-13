# Description
Coming soon. 

Full data is available under `[https://github.com/gucorpling/amalgum/tree/master/full](full/)`. 

**NOTE**: `xml/` and `dep/` are finished, but `rst/` and `tsv/` are still coming. (ETA: 3/20/2020.) 

# Build
## Dependencies
1. Set up a conda environment:

```sh
conda create --name amalgum python=3.7
conda activate amalgum
pip install -r requirements.txt
```

2. Install a version of `pytorch` and `tensorflow` 1.x appropriate for your machine's hardware.

3. Download [files required for GUMDROP](https://corpling.uis.georgetown.edu/amir/gumdrop/) and follow
[the instructions](https://corpling.uis.georgetown.edu/amir/gumdrop/README.md) for unpacking them.

4. Download `punkt`: `python -c "import nltk; nltk.download('punkt')"`

5. Download [the UDPipe binary](https://github.com/ufal/udpipe/releases/download/v1.2.0/udpipe-1.2.0-bin.zip) bundle and move the binary that is appropriate for your machine to `lib/gumdrop/lib/udpipe/udpipe`

## NLP Pipeline
Invoke nlp_controller.py on the tiny subset to ensure the pipeline is working properly:

```bash
python nlp_controller.py target -i out_tiny
```

# Development

## Formatting
**Be sure** to set up black:

```sh
# install pre-commit formatting hooks
pre-commit install
```

We use [black](https://github.com/psf/black) in this repository. Black is a program that automatically formats your Python code. When you typed `pre-commit install` above, you told git to run `black` according to the settings in `.pre-commit-config.yaml` every time there's a relevant commit.

When you commit, if black decided to reformat any of your files, the commit will fail. At this point:
1. If you are using PyCharm, all you will need to do is review the files that were changed, make sure black did an OK job, then commit again.
2. If you are using `git` directly in a console, you will need to re-`git add` your files and do the commit again.
