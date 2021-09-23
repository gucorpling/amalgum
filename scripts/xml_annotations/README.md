## Clean up of xml files

The amalgum xml files have been altered in the following ways: 
- Several tags have been removed ('dd', 'dl', 'dt', 'tbody')
- Several tags have been mapped to different tag names (see details in ```reduce_xml_tags``` function)
- Redudant attributes in nested tags have been removed
- Empty tags that span no text and carry no additional information have been removed
- All 'ref' tags within a sentence that span no tokens have been moved to cover the token that precedes it 

The code for these alterations can be found in the ```reduce_amalgum_xml.py``` file.

### manual corrections

The following files also required manual correction:
- AMALGUM_bio_editions.xml - token order correction starting line 926
- AMALGUM_news_strip.xml - correction of token order to match conllu file, starting at line 407
- AMALGUM_bio_paulides - correction of token order to match conllu file, starting at line 1460
- AMALGUM_interview_shiflett - correction of token order to match conllu file, starting at line 200
- AMALGUM_bio_hatuey - correction of token order to match conllu file, starting at line 97
- AMALGUM_interview_zille - correction of token order to match conllu file, starting at line 478
- AMALGUM_bio_semba - correction of token order to match conllu file, starting at line 1489
- AMALGUM_reddit_audible.xml: typo correction / -> ) on line 306
- AMALGUM_interview_incentives - Nested speaker tags that span no text have been made sequential 
- AMALGUM_voyage_chiapas - a 'ref' tag on line 790 that spanned no text was moved to cover a token
- AMALGUM_voyage_gotemba - a final file 'ref' tag was moved to be associated with the last sentence

## xml annotations in conllu

The ```add_xml_annotations.py``` file has functions that take an xml file and the corresponding conllu file and write a new conllu file with xml annotations added.  

In the updated conllu files, there are two places where you'll find xml annotations. One is as sentence level comments, and the other is as an entry in the misc. column of the tokens. Here is an example of a sentence level comment:

```# newpar = p (4 s)```

This indicates that there is a ```<p>``` tag opening before the start of the sentence that it is a comment on, and that the tag spans 4 sentences before closing. If a sentence has multiple xml comments, the tags open in the order they appear in the comments from top to bottom. 

Here is a sample xml annotation from the misc. column of a token: 

```XML=<date when:::"1520"></date>```

The misc. column annotation lists (in order) the tags that open directly before this token and the tags that close directly after this token. In this case, the date tag surrounds just the one token.

There is also a special sentence level comment specifically for tags that trail directly after a sentence, not spanning any sentences or tokens (these are mainly figures):

```# trailing_xml = <figure rend:::"Oil Drum.png"></figure>```

