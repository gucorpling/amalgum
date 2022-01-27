**Note 2022-01-26**: This code is brittle and you should prefer using [wiki-thresher](https://github.com/lgessler/wiki-thresher) for new scraping.
Code should remain here for historical purposes.

# Usage
1. Make sure you've installed the requirements from `../requirements.txt`

2. Install [Node.js](https://nodejs.org/en/)

3. Install [npm](https://docs.npmjs.com/cli/install)

4. Set up parsoid:

```sh
# make sure you checked out the parsoid submodule
git submodule update
cd parsoid
npm install
cd ..
```

5. Select a config describing the MediaWiki instance you want to scrape, e.g. `configs/wikinews.yaml`

6. Use `main.py` to prototype with a single file:

```sh
python main.py \
--config configs/wikinews.yaml \
--method url \
--url 'https://en.wikinews.org/wiki/%22Avast_ye_scurvy_file_sharers!%22:_Interview_with_Swedish_Pirate_Party_leader_Rickard_Falkvinge' \
| tee tmp.xml
# inspect the output and make changes to the yaml config as necessary
google-chrome tmp.xml
```

7. Once you're satisfied with the config, begin scraping:

```sh
python main.py --method scrape --config configs/wikinews.yaml --output-dir output/wikinews
```

# Notes
- If an article has already been encountered, it will not be pulled again **regardless** of whether it has been updated since the last time it was seen
- GUM TEI output will always be generated anew for all files (regardless of whether or not its mwtext was cached) according to the config as it currently is.
