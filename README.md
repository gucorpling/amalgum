# Development

## Setup
Using conda:

```sh
conda create --name gumby python=3.7
conda activate gumby
pip install -r requirements.txt
```

**Be sure** to set up black:

```sh
# install pre-commit formatting hooks
pre-commit install
```

## Formatting
We use [black](https://github.com/psf/black) in this repository. Black is a program that automatically formats your Python code. When you typed `pre-commit install` above, you told git to run `black` according to the settings in `.pre-commit-config.yaml` every time there's a relevant commit.

When you commit, if black decided to reformat any of your files, the commit will fail. At this point:
1. If you are using PyCharm, all you will need to do is review the files that were changed, make sure black did an OK job, then commit again.
2. If you are using `git` directly in a console, you will need to re-`git add` your files and do the commit again.
