## Clean up of xml files

Before being put into the conllu files as xml annnotations, the amalgum xml data was altered in the following ways: 
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

```# newpar = sp who:::"#PsychologicalCable" (36 s) | p (36 s)```

This comment holds information for 2 different tags, separated by ```|```. This indicates that there is a ```<sp>``` tag opening before the start of the sentence that it is a comment on, that the tag as the attribute ```who=#PsychologicalCable```, and that the tag spans 36 sentences before closing. There is also a ```<p>``` tag following the ```<sp>``` tag that opens before the start of the sentence, and that tag also spans 36 sentences. The information for different tags is separated by ```|```, and the tags open in the order they appear in the comment from left to right.

Here is a sample xml annotation from the misc. column of a token: 

```XML=<date when:::"1520"></date>```

The misc. column annotation lists (in order) the tags that open directly before this token and the tags that close directly after this token. In this case, the date tag surrounds just the one token. (Note that the typical ```=``` for xml attributes has been replaced with ```:::``` within the annotations)

There is also a special sentence level comment specifically for tags that trail directly after a sentence, not spanning any sentences or tokens (these are mainly figures):

```# trailing_xml = <figure rend:::"Oil Drum.png"></figure>```