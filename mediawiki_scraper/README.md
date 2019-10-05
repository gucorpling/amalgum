# Usage
1. Make sure you've installed the requirements from `../requirements.txt`

2. Install [Node.js](https://nodejs.org/en/)

3. Install [npm](https://docs.npmjs.com/cli/install)

4. Set up parsoid:

```sh
# make sure you checked out the parsoid submodule
git submodule update --recursive
cd parsoid
npm install
cd ..
```

4. Select a config describing the MediaWiki instance you want to scrape, e.g. `configs/wikinews.yaml`

5. Use `main.py` to prototype with a single file:

```sh
python main.py \
--config configs/wikinews.yaml \
--method file \
--file mwtext/https-en-wikinews-org-wiki-22avast-ye-scurvy-file-sharers-21-22-3a-interview-with-swedish-pirate-party-leader-rickard-falkvinge_3911634  \
| tee tmp.xml
# inspect the output and make changes to the yaml config as necessary
google-chrome tmp.xml
```

6\. Once you're satisfied with the config, begin scraping:

```sh
python main.py --method scrape --config configs/wikinews.yaml
```

7\. Output will be available in the `gum_tei` directory.
