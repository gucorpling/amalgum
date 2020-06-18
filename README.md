# AMALGUM v0.1
## Download
Latest data **without Reddit texts** for is available under [`amalgum/`](https://github.com/gucorpling/amalgum/tree/master/amalgum) and [`amalgum_balanced`](https://github.com/gucorpling/amalgum/tree/master/amalgum_balanced). (The `_balanced` variant contains nearly 500,000 tokens for each genre, while the unbalanced variant contains slightly more data.)

You may [download the data **without Reddit texts** as a zip](https://drive.google.com/file/d/1StyZjJ6u84vZgJ2bIgsuCb037zc36RXB/view?usp=sharing). The complete corpus, with Reddit data, is available upon request: please email [lg876@georgetown.edu](mailto:lg876@georgetown.edu).

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
* (Non-)named entities 
* Coreference
* Rhetorical structure theory

These annotations are across four file formats: [GUM-style XML](https://github.com/amir-zeldes/gum), [CONLLU](https://universaldependencies.org/format.html), [WebAnno TSV](https://webanno.github.io/webanno/releases/3.4.5/docs/user-guide.html#sect_webannotsv), and [RS3](https://github.com/gucorpling/rst-xsd).

You can see samples of the data for `bio_doc124`: [xml](https://github.com/gucorpling/amalgum/blob/master/amalgum/xml/amalgum_bio_doc124.xml), [conllu](https://github.com/gucorpling/amalgum/blob/master/amalgum/dep/amalgum_bio_doc124.conllu), [tsv](https://github.com/gucorpling/amalgum/blob/master/amalgum/tsv/amalgum_bio_doc124.tsv), [rs3](https://github.com/gucorpling/amalgum/blob/master/amalgum/rst/amalgum_bio_doc124.rs3)

### Further Information
Please see [our paper](https://www.aclweb.org/anthology/2020.lrec-1.648.pdf).

# License
All annotations under the folders `amalgum/` and `amalgum_balanced/` are available under a [Creative Commons Attribution (CC-BY) license, version 4.0](https://creativecommons.org/licenses/by/4.0/). Note that their texts are sourced from the following websites under their own licenses:

* `academic`: [MDPI](https://www.mdpi.com/about), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* `bio`: [Wikipedia](https://creativecommons.org/licenses/by/4.0/), [CC BY-SA 3.0](https://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License)
* `fiction`: [Project Gutenberg](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License), [The Project Gutenberg License](https://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License#The_Full_Project_Gutenberg_License_in_Legalese_.28normative.29)
* `interview`: [Wikinews](https://en.wikinews.org/wiki/Wikinews:Copyright), [CC BY 2.5](http://creativecommons.org/licenses/by/2.5/)
* `news`: [Wikinews](https://en.wikinews.org/wiki/Wikinews:Copyright), [CC BY 2.5](http://creativecommons.org/licenses/by/2.5/)
* `whow`: [wikiHow](https://www.wikihow.com/), [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/)
* `voyage`: [wikiVoyage](https://en.wikivoyage.org/wiki/Wikivoyage:Dual_licensing), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

# Citation

**NOTE:** Our [ACL Anthology](https://www.aclweb.org/anthology/2020.lrec-1.648/) listing currently uses an old name for the corpus. **Please use this corrected citation:** 

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

# Build
See [BUILD.md](./BUILD.md).
