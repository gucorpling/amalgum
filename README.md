# AMALGUM v0.2

AMALGUM is a machine annotated multilayer corpus following the same design and annotation layers as [GUM](https://github.com/amir-zeldes/gum), but substantially larger (around 4M tokens). 
The goal of this corpus is to close the gap between high quality, richly annotated, but small datasets, and the larger but shallowly annotated corpora that are often scraped from the Web.
Read more here: https://corpling.uis.georgetown.edu/gum/amalgum.html

## Download

Latest data **without Reddit texts** is available under [`amalgum/`](https://github.com/gucorpling/amalgum/tree/master/amalgum) and some additional data beyond the target size of 4M tokens [`amalgum_extra/`](https://github.com/gucorpling/amalgum/tree/master/amalgum_extra). (The `amalgum` directory contains around 500,000 tokens for each genre, while the extra directory contains some more data beyond the genre-balanced corpus.)

You may [download the older version 0.1 of the corpus **without Reddit texts** as a zip](https://drive.google.com/file/d/1StyZjJ6u84vZgJ2bIgsuCb037zc36RXB/view?usp=sharing). The complete corpus, with Reddit data, is available upon request: please email [lg876@georgetown.edu](mailto:lg876@georgetown.edu).

## Description
AMALGUM (**A** **M**achine-**A**nnotated **L**ookalike of [**GUM**](https://github.com/amir-zeldes/gum)) is an English web corpus spanning 8 genres with 4,000,000 tokens and several annotation layers.

### Genres
Source data was scraped from eight different sources containing stylistically distinct text. Each text's source is indicated with a slug in its filename:

* `academic`: [MDPI](https://www.mdpi.com)
* `bio`: [Wikipedia](http://en.wikipedia.org)
* `fiction`: [Project Gutenberg](https://www.gutenberg.org)
* `interview`: [Wikinews, Interview category](https://en.wikinews.org/wiki/Category:Interview)
* `news`: [Wikinews](https://en.wikinews.org)
* `reddit`: [Reddit](https://www.reddit.com)
* `whow`: [wikiHow](https://www.wikihow.com)
* `voyage`: [wikiVoyage](https://en.wikivoyage.org)

### Annotations
AMALGUM contains annotations for the following information:

* Tokenization
* [UD](https://universaldependencies.org/u/pos/) and [Extended PTB](https://corpling.uis.georgetown.edu/ptb_tags.html) part of speech tags
* Lemmas
* [UD](https://universaldependencies.org/u/dep/) dependency parses
* (Non-)named nested entities
* Coreference resolution
* Rhetorical Structure Theory discourse parses (constituent and dependency versions)
* Date/Time annotations in TEI format

These annotations are across four file formats: [GUM-style XML](https://github.com/amir-zeldes/gum), [CONLLU](https://universaldependencies.org/format.html), [WebAnno TSV](https://webanno.github.io/webanno/releases/3.4.5/docs/user-guide.html#sect_webannotsv), and [RS3](https://github.com/gucorpling/rst-xsd).

You can see samples of the data for `AMALGUM_news_khadr`: [xml](https://github.com/gucorpling/amalgum/blob/master/amalgum/news/xml/AMALGUM_news_khadr.xml), [conllu](https://github.com/gucorpling/amalgum/blob/master/amalgum/news/dep/AMALGUM_news_khadr.conllu), [tsv](https://github.com/gucorpling/amalgum/blob/master/amalgum/news/tsv/AMALGUM_news_khadr.tsv), [rs3](https://github.com/gucorpling/amalgum/blob/master/amalgum/news/rst/AMALGUM_news_khadr.rs3)

### Performance

Current scores on the GUM corpus test set per task:

| task | metric | performance |
| ---- | ----- | ---------- |
| tokenizer | F1 | 99.92 |
| sentencer | Acc / F1 | 99.85 / 94.35 |
| xpos | Acc | 98.16 |
| dependencies | LAS / UAS* | 92.16 / 94.25 |
| NNER | Micro F1 | 70.8 |
| coreference | CoNLL F1 | 51.4 |
| RST | S / N / R | 77.98 / 61.79 / 44.07 |

\* Parsing scores ignore punctuation attachment; punctuation is attached automatically via [udapi](https://udapi.github.io/).

### Further Information
Please see [our paper](https://www.aclweb.org/anthology/2020.lrec-1.648.pdf).

# Citation

```
@inproceedings{gessler-etal-2020-amalgum,
    title = "{AMALGUM} {--} A Free, Balanced, Multilayer {E}nglish Web Corpus",
    author = "Gessler, Luke  and
      Peng, Siyao  and
      Liu, Yang  and
      Zhu, Yilun  and
      Behzad, Shabnam  and
      Zeldes, Amir",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.648",
    pages = "5267--5275",
    abstract = "We present a freely available, genre-balanced English web corpus totaling 4M tokens and featuring a large number of high-quality automatic annotation layers, including dependency trees, non-named entity annotations, coreference resolution, and discourse trees in Rhetorical Structure Theory. By tapping open online data sources the corpus is meant to offer a more sizable alternative to smaller manually created annotated data sets, while avoiding pitfalls such as imbalanced or unknown composition, licensing problems, and low-quality natural language processing. We harness knowledge from multiple annotation layers in order to achieve a {``}better than NLP{''} benchmark and evaluate the accuracy of the resulting resource.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
```


# License
All annotations under the folders `amalgum/` and `amalgum_extra/` are available under a [Creative Commons Attribution (CC-BY) license, version 4.0](https://creativecommons.org/licenses/by/4.0/). Note that their texts are sourced from the following websites under their own licenses:

* `academic`: [MDPI](https://www.mdpi.com/about), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* `bio`: [Wikipedia](https://creativecommons.org/licenses/by/4.0/), [CC BY-SA 3.0](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License)
* `fiction`: [Project Gutenberg](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License), [The Project Gutenberg License](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License#The_Full_Project_Gutenberg_License_in_Legalese_.28normative.29)
* `interview`: [Wikinews](https://en.wikinews.org/wiki/Wikinews:Copyright), [CC BY 2.5](http://creativecommons.org/licenses/by/2.5/)
* `news`: [Wikinews](https://en.wikinews.org/wiki/Wikinews:Copyright), [CC BY 2.5](http://creativecommons.org/licenses/by/2.5/)
* `whow`: [wikiHow](https://www.wikihow.com/), [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* `voyage`: [wikiVoyage](https://en.wikivoyage.org/wiki/Wikivoyage:Dual_licensing), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)


# Development
See [DEVELOPMENT.md](./DEVELOPMENT.md).
