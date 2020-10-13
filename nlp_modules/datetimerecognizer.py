import jpype
import xml.etree.ElementTree as ET # this is fast!
import html
import re
import pickle
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

from nlp_modules.configuration import XML_ATTRIB_REFDATE,XML_ROOT_TIMEX3, DATE_FILTER_PROBA_THRESHOLD
from nlp_modules.base import NLPModule
from glob import glob


class HeidelTimeWrapper():

    """
    Wrapper around java-based HeidelTime - uses jpype to manage JVM
    Thanks to : https://github.com/amineabdaoui/python-heideltime.git
    This speeds up per-file processing in the JVM by about 5 times versus an os.process based call
    (roughly 1 minute for the full GUM corpus v6.0)
    """

    def __init__(self, lang, doc=None, output=None,jarpath=None):

        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % jarpath,convertStrings=True)

        # get the Java classes we want to use
        heideltime_resources = jpype.JPackage("de.unihd.dbs.uima.annotator.heideltime.resources")
        heideltime_standalone = jpype.JPackage("de.unihd.dbs.heideltime.standalone")

        # constants
        LANGUAGES = {
            'english': heideltime_resources.Language.ENGLISH
            #'german': heideltime_resources.Language.GERMAN,
            #'dutch': heideltime_resources.Language.DUTCH,
            #'italian': heideltime_resources.Language.ITALIAN,
            #'spanish': heideltime_resources.Language.SPANISH,
            #'arabic': heideltime_resources.Language.ARABIC,
            #'french': heideltime_resources.Language.FRENCH,
            #'chinese': heideltime_resources.Language.CHINESE,
            #'russian': heideltime_resources.Language.RUSSIAN,
            #'portuguese': heideltime_resources.Language.PORTUGUESE
        }

        DOCUMENTS = {
            'narratives': heideltime_standalone.DocumentType.NARRATIVES,
            'news': heideltime_standalone.DocumentType.NEWS,
            'colloquial': heideltime_standalone.DocumentType.COLLOQUIAL,
            'scientific': heideltime_standalone.DocumentType.SCIENTIFIC
        }

        OUTPUTS = {
            'timeml': heideltime_standalone.OutputType.TIMEML,
            'xmi': heideltime_standalone.OutputType.XMI
        }

        CONFIG = '/home/gooseg/Desktop/heideltime-standalone-2.2.1/heideltime-standalone/config.props'

        self.language = LANGUAGES[lang]
        if (doc is None):
            self.doc_type = DOCUMENTS['news']
        else:
            self.doc_type = DOCUMENTS[doc]
        if (output is None):
            self.output_type = OUTPUTS['timeml']
        else:
            self.output_type = OUTPUTS[output]
        self.heideltime = heideltime_standalone.HeidelTimeStandalone(self.language, self.doc_type, self.output_type, CONFIG)

    def convert_date(self, day, month, year):
        sdf = jpype.java.text.SimpleDateFormat('dd-M-yyyy hh:mm:ss')
        str_date = str(day)+'-'+str(month)+'-'+str(year)+' 00:00:00'
        return sdf.parse(str_date)

    def parse(self, text, date_ref=None):
        """
        date_ref format must be YYYY-MM-DD or exception
        this may be guaranteed if dateCreated in the xml is YYYY-MM-DD
        """
        if (date_ref is None):
            document_creation_date = jpype.java.util.Date()
        else:
            # convert to Java.util.Date
            document_creation_date = self.convert_date(date_ref.split('-')[2],date_ref.split('-')[1],date_ref.split('-')[0])

        # the main juice...ensure that convertString param is set to True when starting the JVM or you'll get a java String here
        return self.heideltime.process(text, document_creation_date)

class DateTimeFilterModel():
    def __init__(self,modelfile=None,datafile=None):
        """
        modelfile - path to the pickle file used as the filter
        datafile - path to the training data file.
        """

        self.articles=['a','an']

        #load the random forest model
        with open('/home/gooseg/Desktop/datetimeparsers/datetimeparsers/evaluation/data/datetimefilter.pickle', 'rb') as f:
            self.rf = pickle.load(f)


        # this is just a template and not the real feature set; this is used to build the real featureset
        # the postags are taken from the label encoder model that encodes the result of the
        # pos tag ensembler
        self.featuredict = {'obl': 0, 'obl:npmod': 0, 'obl:tmod': 0, 'nsubj': 0, 'nsubj:pass': 0, 'obj': 0, 'iobj': 0,
                       'csubj': 0, 'csubj:pass': 0, 'ccomp': 0, 'xcomp': 0, 'nummod': 0, 'acl': 0, 'amod': 0,
                       'appos': 0, 'acl:relcl': 0, 'det': 0, 'det:predet': 0,  'nmod': 0, 'case': 0,
                       'nmod:npmod': 0, 'nmod:tmod': 0, 'nmod:poss': 0, 'advcl': 0, 'advmod': 0, 'neg': 0,
                       'compound': 0, 'compound:prt': 0, 'flat': 0, 'fixed': 0, 'foreign': 0, 'goeswith': 0, 'list': 0,
                       'dislocated': 0, 'parataxis': 0, 'orphan': 0, 'reparandum': 0, 'vocative': 0, 'discourse': 0,
                       'expl': 0, 'aux': 0, 'aux:pass': 0, 'cop': 0, 'mark': 0, 'punct': 0, 'conj': 0, 'cc': 0,
                       'cc:preconj': 0,  'root': 0, 'dep': 0, '$':0, "''":0, ',':0, '-LRB-':0, '-LSB-':0, '-RRB-':0, '-RSB-':0,
                        '.':0, ':':0, 'ADD':0, 'ADJ':0, 'ADP':0, 'ADV':0, 'AFX':0, 'AUX':0, 'CC':0, 'CCONJ':0, 'CD':0,
                        'DET':0, 'DT':0, 'EX':0, 'FW':0, 'GW':0, 'HYPH':0, 'IN':0,'INTJ':0, 'JJ':0, 'JJR':0, 'JJS':0,
                        'LS':0, 'MD':0, 'NFP':0, 'NN':0, 'NNP':0, 'NP':0, 'NNPS':0, 'NNS':0, 'NOUN':0,'NUM':0, 'PART':0, 'PDT':0,
                        'POS':0, 'PRON':0, 'PROPN':0, 'PRP':0, 'PRP$':0, 'PUNCT':0, 'RB':0, 'RBR':0,'RBS':0, 'RP':0, 'SYM':0,
                        'TO':0, 'UH':0, 'VB':0, 'VBD':0, 'VBG':0, 'VBN':0, 'VBP':0, 'VBZ':0, 'VERB':0, 'WDT':0, 'WP':0, 'WP$':0,
                        'WRB':0, 'X':0, '``':0, 'january': 0, 'february': 0, 'march': 0, 'april': 0,
                       'may': 0, 'june': 0, 'july': 0, 'august': 0, 'september': 0, 'october': 0, 'november': 0,
                       'december': 0, 'summer': 0, 'winter': 0, 'autumn': 0, 'spring': 0, 'christmas': 0,
                       'christmas_eve': 0, 'easter': 0, 'easter_sunday': 0, 'monday': 0, 'tuesday': 0, 'wednesday': 0,
                       'thursday': 0, 'friday': 0, 'saturday': 0, 'sunday': 0, 'phrase': None,'sentence_index':None}

    def train(self):
        pass # TODO

    def inference(self):
        pass

    def build_featureset(self,sfull,stok,phrase,index,timextype,timexvalue):
        """
        Builds a row of features for the date phrase from a template and adds some extra features
        sfull is the sentence with each token tagged with its corresponding feature
        stok is just the sentence made up of its english tokens
        phrase is the date phrase we are going to classify
        """

        # Otherwise the split() method wont work well and the indexes will be off
        sfull = re.sub(' +',' ',sfull)
        stok = re.sub(' +', ' ', stok)
        phrase = re.sub(' +',' ',phrase)

        features = sfull.split()
        search = re.search(phrase.lower(),stok.lower()) # this is not supposed to return None, generally

        # Gets the start index of the phrase in the sentence
        if search.start() > 0:
            startindex = stok[0:search.start()].count(' ')
        else:
            startindex = 0

        endindex = startindex + len(phrase.split())

        # start building the features now that we have the indices of the phrase in the sentence
        # Features are basically count vectors of the defined featureset in the template
        fdict = self.featuredict.copy() # need to do deep copy from the template
        featurelist = fdict.keys()

        # This will check features that are specific word tokens in the sentence, e.g January, Tuesday,
        for i in range(startindex,endindex):
            feats = features[i].split('/') # get the token word
            if str(feats[1]).lower() in featurelist:
                fdict[str(feats[1]).lower()] += 1 # Increment if the word is in the feature list
            if str(feats[4]) in fdict.keys():
                fdict[str(feats[4])] += 1 # the Penn treebank POS tag
            if str(feats[7]) in fdict.keys():
                fdict[str(feats[7])] += 1 # the UD tag


        # Next some features based on the immediately preceding and following tokens, padding the phrase
        if startindex > 0:
            prevtokenfeatures = str(features[startindex - 1]).lower().split('/')
            fdict['prev_compound'] = int(prevtokenfeatures[-1].strip() == 'compound')
            fdict['a_an'] = int(prevtokenfeatures[0].lower().strip() in self.articles)
            if 'mod' in prevtokenfeatures[-1]:
                fdict['prev_mod'] = 1
            else:
                fdict['prev_mod'] = 0

            if str(prevtokenfeatures[-1]).strip() == 'dep':
                fdict['prev_dep'] = 1
            else:
                fdict['prev_dep'] = 0

            if 'det' in prevtokenfeatures[-1]:
                fdict['prev_det'] = 1
            else:
                fdict['prev_det'] = 0
        else:
            fdict['a_an'] = 0
            fdict['prev_mod'] = 0
            fdict['prev_compound'] = 0
            fdict['prev_det'] = 0
            fdict['prev_dep'] = 0

        if endindex + 1 < len(features):
            endtokenfeatures = str(features[endindex + 1]).lower().split('/')
            fdict['next_compound'] = int(endtokenfeatures[-1].strip() == 'compound')
            if 'mod' in endtokenfeatures[-1]:
                fdict['end_mod'] = 1
            else:
                fdict['end_mod'] = 0

            if endtokenfeatures[-1].strip() == 'dep':
                fdict['end_dep'] = 1
            else:
                fdict['end_dep'] = 0
            if 'det' in endtokenfeatures[-1]:
                fdict['end_det'] = 1
            else:
                fdict['end_det'] = 0
        else:
            fdict['next_compound'] = 0
            fdict['end_mod'] = 0
            fdict['end_det'] = 0
            fdict['end_dep'] = 0

        # and finally, dump out all the other interactions
        fdict['CD_nmod'] = int(fdict['CD']) * int(fdict['nmod'])
        fdict['CD_nmodtmod'] = int(fdict['CD']) * int(fdict['nmod:tmod'])
        fdict['CD_compound'] = int(fdict['CD']) * int(fdict['compound'])
        fdict['CD_nummod'] = int(fdict['CD']) * int(fdict['nummod'])

        fdict['countmods'] = fdict['obl:npmod'] + fdict['obl:tmod'] + fdict['nummod'] + fdict['amod'] + fdict['nmod'] + \
                             fdict['nmod:npmod'] + fdict['nmod:tmod'] + fdict['nmod:poss'] + fdict['advmod']

        fdict['summer_compound'] = fdict['summer'] * fdict['compound']
        fdict['winter_compound'] = fdict['winter'] * fdict['compound']
        fdict['autumn_compound'] = fdict['autumn'] * fdict['compound']
        fdict['spring_compound'] = fdict['spring'] * fdict['compound']

        fdict['easter_compound'] = fdict['easter'] * fdict['compound']
        fdict['eastersunday_compound'] = fdict['easter_sunday'] * fdict['compound']
        fdict['christmas_compound'] = fdict['christmas'] * fdict['compound']
        fdict['christmaseve_compound'] = fdict['christmas_eve'] * fdict['compound']

        fdict['CD_obl'] = fdict['CD'] * fdict['obl']
        fdict['CD_dep'] = fdict['CD'] * fdict['dep']
        fdict['sunday_obl'] = fdict['sunday'] * fdict['obl']
        fdict['monday_obl'] = fdict['monday'] * fdict['obl']
        fdict['tuesday_obl'] = fdict['tuesday'] * fdict['obl']
        fdict['wednesday_obl'] = fdict['wednesday'] * fdict['obl']
        fdict['thursday_obl'] = fdict['thursday'] * fdict['obl']
        fdict['friday_obl'] = fdict['friday'] * fdict['obl']
        fdict['saturday_obl'] = fdict['saturday'] * fdict['obl']

        fdict['NP_flat'] = fdict['NP'] * fdict['flat']
        fdict['NP_obl'] = fdict['NP'] * fdict['obl']
        fdict['NP_nummod'] = fdict['NP'] * fdict['nummod']
        fdict['NP_nmod'] = fdict['NP'] * fdict['nmod']
        fdict['NP_compound'] = fdict['NP'] * fdict['compound']
        fdict['NN_compound'] = fdict['NN'] * fdict['compound']
        fdict['RB_amod'] = fdict['RB'] * fdict['amod']
        fdict['JJ_amod'] = fdict['JJ'] * fdict['amod']

        # these become useful when adding the tags to the xml
        fdict['sentence_index'] = index + 1
        fdict['start_index'] = int(startindex)
        fdict['phrase'] = phrase
        fdict['timextype'] = timextype
        fdict['timexvalue'] = timexvalue

        """
        TODO: bring these back , use the file name which contains the category 
        fdict['news'] = int(category == 'news')
        fdict['interview'] = int(category == 'interview')
        fdict['bio'] = int(category == 'bio')
        """

        return fdict


class DateTimeRecognizer(NLPModule):
    def __init__(self,heideltimeobj,datefilterobj,postaglabelencoderobj,decoding='ascii'):

        super().__init__(config=None)
        self.decoding = decoding
        self.hw = heideltimeobj
        self.datefilter = datefilterobj

        self.regex1 = r'[^0-9-]' # anything not a number or hyphen
        self.regex2 = r'[0-9]{4}-[0-9]{2}-[0-9]{2}' # matches YYYY-MM-DD
        self.regex3 = r'XXXX-[0-9]{2}-[0-9]{2}' #--mm-dd
        self.regex4 = r'XXXX-XX-[0-9]{2}'  # matches --dd
        self.regex5 = r'XXXX-[0-9]{2}-XX'  # matches --mm
        self.regex6 = r'[0-9]{4}-[0-9]{2}'  # matches YYYY-MM
        self.regex7 = r'[0-9]{4}'  # matches YYYY
        self.seasons = {'SU':['--06','--09'],'WI':['--12','--03'],'FA':['--09','--12'],'SP':['--03','--06']}
        self.holidates = {'spanish golden age':['from:1556','to:1659'],'easter':['notBefore:--03','notAfter:--05'],'easter sunday':['notBefore:--03','notAfter:--05'],'christmas':['when:--12-25'],'christmas eve':['when:--12-24'],'world war 2':['from:1939-09-01','to:1945-02-01'],'world war ii':['from:1939-09-01','to:1945-02-01'],'world war 1':['from:1914','to:1918'],'world war i':['from:1914','to:1918'],'the american revolution':['notBefore:1775','notAfter:1783'],'the american revolutionary war':['notBefore:1775','notAfter:1783'],'the civil war':['notBefore:1861','notAfter:1865'],'the american civil war':['notBefore:1861','notAfter:1865'],'the reconstruction era':['notBefore:1863','notAfter:1887']}

        # TODO: 'soft-wire' feature names from the label encoder for pos tagging instead of hard-wiring in the dictionary features
        #self.cd = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        #with open(self.cd + postaglabelencoderobj, "rb") as f:
        #    le = pickle.load(f)

    def requires(self):
        pass

    def provides(self):
        pass

    def annotate_xml_with_date(self,datephrase,sentenceindex,timextype,timexvalue,xmltreeobj):
        # TODO:
        # YYYY-MM-DD and all its variations
        # gazette for listed holidayes
        # seasons
        # from-to or notbefore-notafter
        # centuries, decades
        # times

        timexvalue = re.sub(self.regex1,'X',timexvalue)

        pass

    def test_dependencies(self):
        """
        TODO: copy heideltime standalone and treetagger to the bin folder and Runs the test by trying to parse an example sentence
        """
        pass

    def process_file(self,filename):
        """
        Method to process a single file
        :param filename: The conllu filename to parse
        :return: nada (writes new xml with date tags to the folder step)
        """

        def add_datetime_tags(node,counter=0):
            """
            Recursively builds the new xml file in memory, and stamps the date xml on it
            don't remove the counter, it is an accumulator that keep tracks of how many sentences we have iterated over
            """
            if node is not None:
                for item in node:
                    if item.tag == 's':
                        counter += 1
                        if counter in sentenceindices:
                            df = indexphrases.loc[indexphrases['sentence_index'] == counter]
                            datetags = [] # only way to add the tags is sequentially after determining the text and tails
                            for _,row in df.iterrows(): # adding date tags backwards up
                                phrase = str(row['phrase'])
                                startindex = int(row['start_index'])
                                endindex = startindex + len(phrase.split())

                                splittext = item.text.split('\n')
                                splittext = [t for t in splittext if t]
                                predatetext = splittext[0:startindex]
                                datetext = splittext[startindex:endindex]
                                postdatetext = splittext[endindex:len(splittext)]

                                # build the xml elements and date tags
                                if str('\n'.join(predatetext)).strip() == '':
                                    item.text = '\n'
                                else:
                                    item.text = '\n' + '\n'.join(predatetext) + '\n'

                                date = ET.Element('date') # TODO: build attributes and time tags
                                date.text = '\n' + '\n'.join(datetext) + '\n'

                                if str('\n'.join(postdatetext)).strip() == '':
                                    date.tail = '\n'
                                else:
                                    date.tail = '\n' + '\n'.join(postdatetext) + '\n'

                                datetags.append(date)

                            # now build the final tag sequentially with all nested date tags
                            for i in range(len(datetags) - 1,-1,-1):
                                item.append(datetags[i])

                    counter = add_datetime_tags(item,counter) # don't remove the accumulator pattern
            else:
                return 0

            return counter

        """
        def make_sent(text):
            
            #Helper method to get the sentence with features
            #:param text: text rows to parse
            #:return: tokens from the text rows aligned in sentence format
            
            sents = text.split('\n')
            sents = [sent.replace('\t', '/') for sent in sents] # all features seprated by '/'
            senttokens = [sent.split('/')[0] for sent in sents]
            sents = ' '.join(sents)
            senttokens = ' '.join(senttokens)
            return sents.strip(), senttokens.strip()
        """

        # The POS tags and UD tags are in the conllu format..
        conllufile = '/'.join(filename.split('/')[0:-2]) + '/dep/' + filename.split('/')[-1].replace('.xml','.conllu')
        print(conllufile)

        xmltree = ET.parse(filename)
        root = xmltree.getroot()

        dateCreated = root.attrib[XML_ATTRIB_REFDATE] # assumed to be in default YYYY-MM-DD or the process breaks

        sentences = [] # to hold the list of sentences in the file built from everything
        sentencestokens = [] # holds sentences built from tokens only

        # build the sentences from the conllu file
        # sentence boundaries have multiple \n in between
        with open(conllufile,'r') as r:
            sent = []
            senttok = []
            for line in r:
                line = line.strip()
                if line == '':
                    if len(sent) == 0: continue # second newline
                    sentences.append(sent)
                    sentencestokens.append(senttok)
                    sent = []
                    senttok = []
                else:
                    senttok.append(line.split('\t')[1])
                    sent.append(line.replace('\t','/')) # changes the delimiter to a cleaner one

        """
        # old way - build from XML 
        for sent in xmltree.iter('s'):

            sent, senttoken = make_sent(ET.tostring(sent, method='text').decode(self.decoding))

            # some html and xml unfriendly characters throw off processing..
            senttoken = html.unescape(senttoken)
            senttoken = self.replace_xml_chars(senttoken)

            sentences.append(sent)
            sentencestokens.append(senttoken)
        """

        # rollup
        for i in range(0,len(sentencestokens)):
            sentencestokens[i] = str(' '.join(sentencestokens[i])).strip()
            sentences[i] = str(' '.join(sentences[i])).strip() # now inner delimited by '/

        # now call heideltime to process the whole file at once
        text = '\n'.join(sentencestokens)
        # the main course...you should have had your appetizers by now..
        result = self.hw.parse(text,dateCreated) # heideltime lets you pass a reference date for each file. #TODO: pass category e.g 'news'

        # We'll assume here that the list of dates returned matches the sentence indices properly 1-1.
        # This has been empirically proven.
        dates,attribs = self.parse_timex3_xml(result)

        # build dataframe for inference
        # there are as many number of dates as there are number of sentences
        # so the prediction can be made for the whole file at once
        inferencedf = pd.DataFrame(columns=self.datefilter.featuredict.keys())
        for i in range(0,len(dates)):
            if str(dates[i]).strip() == '': continue # nothing

            # each date in the sentence is delimited by semi-colon, as there might be more than one date detected in a sentence
            # also build the attributeset
            subdates = dates[i].split(';')
            for j in range(0,len(subdates)):
                timextype = attribs[i][j][0]['type']
                timexvalue = attribs[i][j][0]['value']
                f = self.datefilter.build_featureset(sentences[i],sentencestokens[i],subdates[j],i,timextype,timexvalue)
                inferencedf = inferencedf.append(f,ignore_index=True)

        indexphrases = inferencedf[['sentence_index','start_index','phrase','timextype','timexvalue']]
        inferencedf.drop(columns=['sentence_index','start_index','phrase','timextype','timexvalue'],axis=1,inplace=True)

        # Filter the dates that dont pass GUM annotated standards..
        tpprobs = self.datefilter.rf.predict_proba(inferencedf)[:,1]
        tpprobs = (tpprobs > DATE_FILTER_PROBA_THRESHOLD).astype(int).tolist() # 0's or 1's based on the threshold

        indexphrases['label'] = pd.Series(tpprobs)
        indexphrases = indexphrases.loc[indexphrases['label'] == 1]
        if indexphrases is not None and len(indexphrases) != 0: # only if we have dates..
            indexphrases.sort_values(['sentence_index','start_index'],ascending=[True,False],inplace=True)

            # just in case..this leads to selection of only 1 date element in the same token....
            indexphrases= indexphrases.groupby(by=['sentence_index','start_index']).head(1)
            sentenceindices = set(indexphrases['sentence_index'].tolist())

            # Build the xml with the new date tag
            _ = add_datetime_tags(root) # modify xml in place and add date tags

        # write to disk
        tree = ET.ElementTree(root)
        tree.write(open('test.xml', 'w'), encoding='unicode',xml_declaration=True)

    def replace_xml_chars(self,text):
        return text.replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;').replace('"', '&quot;').replace("'",'&apos;')  # assumes these dont affect time recognition..

    def get_attr(self, xml):
        attributes = []
        if len(xml.attrib) != 0:
            attributes.append(xml.attrib)
        return attributes

    def parse_timex3_xml(self, xmltext):
        """
        Parses the TimeX3 file and returns hypotheses separated by ';'
        :param xmlfile: The xml file output with TIMEX3 tags created by the tool
        :return: list of dates for each sentence separated by ';'
        """
        xmltree = ET.fromstring(xmltext)

        # extract the sentences
        for node in xmltree.iter(XML_ROOT_TIMEX3):
            s = html.unescape(ET.tostring(node, method='text').decode(self.decoding, 'ignore'))

        s = s.split('\n')
        s = [t for t in s if t]  # removes extra spaces

        # fetch all the dates
        all_dates = []
        all_attributes = []
        for node in xmltree.iter('TIMEX3'):
            attributes = self.get_attr(node)
            if len(attributes) > 0:
                if attributes[0]['type'] == "SET": continue # disregard sets entirely from being annotated.

            if '\n' in node.text: continue  # invalid date has been tagged
            all_dates.append(node.text)
            all_attributes.append(attributes)

        # Now 'assign' the date to the text sentence
        i = 0
        j = 0

        sorted_dates = []
        sorted_attribs = []
        dates = []
        attribs = []
        while i < len(s):
            if j == len(all_dates):
                sorted_dates.append(dates)
                sorted_attribs.append(attribs)
                break

            if all_dates[j] in s[i]:
                dates.append(all_dates[j])
                attribs.append(all_attributes[j])
                j += 1
            else:
                sorted_dates.append(dates)
                sorted_attribs.append(attribs)
                dates = []
                attribs = []
                i += 1

        sorted_dates = [';'.join(date) for date in sorted_dates]
        sorted_attribs = [None if len(s) == 0 else s for s in sorted_attribs]

        return sorted_dates,sorted_attribs

    def run(self, input_dir, output_dir):

        # Get list of all xml files to parse
        for file in glob(input_dir + '*.xml'):
            self.process_file(file)
            break


def main():
    """
    Testing only
    """

    jar = "/home/gooseg/Desktop/heideltime-standalone-2.2.1/heideltime-standalone/de.unihd.dbs.heideltime.standalone.jar"


    hw = HeidelTimeWrapper('english',jarpath=jar)
    dfilter = DateTimeFilterModel()
    postagleobj = "/nlp_modules/pos-dependencies/all-encodings.pickle.dat"
    dtr = DateTimeRecognizer(heideltimeobj=hw,datefilterobj=dfilter,postaglabelencoderobj=postagleobj)

    start = time.time()
    dtr.run(input_dir='/home/gooseg/Desktop/amalgum/amalgum/target/04_DepParser/xml/',output_dir=None)
    print (time.time() - start)



    pass


if __name__ == "__main__":
    # Testing only
    main()