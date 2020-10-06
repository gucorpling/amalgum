import jpype
import xml.etree.ElementTree as ET # this is fast!
import html
import re

from nlp_modules.configuration import XML_ATTRIB_REFDATE,XML_ROOT_TIMEX3
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

        # this is just a template and not the real feature set; this is used to build the real featureset
        self.featuredict = {'obl': 0, 'obl:npmod': 0, 'obl:tmod': 0, 'nsubj': 0, 'nsubj:pass': 0, 'obj': 0, 'iobj': 0,
                       'csubj': 0, 'csubj:pass': 0, 'ccomp': 0, 'xcomp': 0, 'nummod': 0, 'acl': 0, 'amod': 0,
                       'appos': 0, 'acl:relcl': 0, 'det': 0, 'det:predet': 0,  'nmod': 0, 'case': 0,
                       'nmod:npmod': 0, 'nmod:tmod': 0, 'nmod:poss': 0, 'advcl': 0, 'advmod': 0, 'neg': 0,
                       'compound': 0, 'compound:prt': 0, 'flat': 0, 'fixed': 0, 'foreign': 0, 'goeswith': 0, 'list': 0,
                       'dislocated': 0, 'parataxis': 0, 'orphan': 0, 'reparandum': 0, 'vocative': 0, 'discourse': 0,
                       'expl': 0, 'aux': 0, 'aux:pass': 0, 'cop': 0, 'mark': 0, 'punct': 0, 'conj': 0, 'cc': 0,
                       'cc:preconj': 0,  'root': 0, 'dep': 0, 'AJ0': 0, 'AJC': 0, 'AJS': 0, 'AT0': 0,
                       'AV0': 0, 'AVP': 0, 'AVQ': 0, 'CJC': 0, 'CJS': 0, 'CJT': 0, 'CRD': 0, 'DPS': 0, 'DT0': 0,
                       'DTQ': 0, 'EX0': 0, 'ITJ': 0, 'NN0': 0, 'NN1': 0, 'NN2': 0, 'NP0': 0, 'NULL': 0, 'ORD': 0,
                       'PNI': 0, 'PNP': 0, 'PNQ': 0, 'PNX': 0, 'POS': 0, 'PRF': 0, 'PRP': 0, 'PUL': 0, 'PUN': 0,
                       'PUQ': 0, 'PUR': 0, 'TO0': 0, 'UNC': 0, 'VBB': 0, 'VBD': 0, 'VBG': 0, 'VBI': 0, 'VBN': 0,
                       'VBZ': 0, 'VDB': 0, 'VDD': 0, 'VDG': 0, 'VDI': 0, 'VDN': 0, 'VDZ': 0, 'VHB': 0, 'VHD': 0,
                       'VHG': 0, 'VHI': 0, 'VHN': 0, 'VHZ': 0, 'VM0': 0, 'VVB': 0, 'VVD': 0, 'VVG': 0, 'VVI': 0,
                       'VVN': 0, 'VVZ': 0, 'XX0': 0, 'ZZ0': 0, 'january': 0, 'february': 0, 'march': 0, 'april': 0,
                       'may': 0, 'june': 0, 'july': 0, 'august': 0, 'september': 0, 'october': 0, 'november': 0,
                       'december': 0, 'summer': 0, 'winter': 0, 'autumn': 0, 'spring': 0, 'christmas': 0,
                       'christmas_eve': 0, 'easter': 0, 'easter_sunday': 0, 'monday': 0, 'tuesday': 0, 'wednesday': 0,
                       'thursday': 0, 'friday': 0, 'saturday': 0, 'sunday': 0, 'label': 0, 'phrase': None}

    def train(self):
        pass # TODO

    def inference(self):
        pass

    def build_featureset(self,sfull,stok,phrase,category):
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
        # and which match the CLAW5 tag and UD tag of the token
        for i in range(startindex,endindex):
            feats = features[i].split('/') # get the token word
            if str(feats[0]).lower() in featurelist:
                fdict[str(feats[0]).lower()] += 1 # Increment if the word is in the feature list
            if str(feats[3]) in fdict.keys():
                fdict[str(feats[3])] += 1 # the CLAW5 tag
            if str(feats[4]) in fdict.keys():
                fdict[str(feats[4])] += 1 # the UD tag


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
        fdict['CRD_nmod'] = int(fdict['CRD']) * int(fdict['nmod'])
        fdict['CRD_nmodtmod'] = int(fdict['CRD']) * int(fdict['nmod:tmod'])
        fdict['CRD_compound'] = int(fdict['CRD']) * int(fdict['compound'])
        fdict['CRD_nummod'] = int(fdict['CRD']) * int(fdict['nummod'])

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

        fdict['CRD_obl'] = fdict['CRD'] * fdict['obl']
        fdict['CRD_dep'] = fdict['CRD'] * fdict['dep']
        fdict['sunday_obl'] = fdict['sunday'] * fdict['obl']
        fdict['monday_obl'] = fdict['monday'] * fdict['obl']
        fdict['tuesday_obl'] = fdict['tuesday'] * fdict['obl']
        fdict['wednesday_obl'] = fdict['wednesday'] * fdict['obl']
        fdict['thursday_obl'] = fdict['thursday'] * fdict['obl']
        fdict['friday_obl'] = fdict['friday'] * fdict['obl']
        fdict['saturday_obl'] = fdict['saturday'] * fdict['obl']

        fdict['NP0_flat'] = fdict['NP0'] * fdict['flat']
        fdict['NP0_obl'] = fdict['NP0'] * fdict['obl']
        fdict['NP0_nummod'] = fdict['NP0'] * fdict['nummod']
        fdict['NP0_nmod'] = fdict['NP0'] * fdict['nmod']
        fdict['NP0_compound'] = fdict['NP0'] * fdict['compound']
        fdict['NN1_compound'] = fdict['NN1'] * fdict['compound']
        fdict['ORD_amod'] = fdict['ORD'] * fdict['amod']

        fdict['news'] = int(category == 'news')
        fdict['interview'] = int(category == 'interview')
        fdict['bio'] = int(category == 'bio')

        return fdict




class DateTimeRecognizer(NLPModule):
    def __init__(self,heideltimeobj,decoding='ascii'):

        super().__init__(config=None)
        self.decoding = decoding
        self.hw = heideltimeobj

    def requires(self):
        pass

    def provides(self):
        pass

    def test_dependencies(self):
        """
        TODO: copy heideltime standalone and treetagger to the bin folder and Runs the test by trying to parse an example sentence
        """
        pass

    def process_file(self,filename):
        """
        Method to process a single file
        :param filename: The xml filename to parse
        :return:
        """

        def make_sent(text):
            """
            Helper method to get the sentence with features
            :param text: text rows to parse
            :return: tokens from the text rows aligned in sentence format
            """
            sents = text.split('\n')
            sents = [sent.replace('\t', '/') for sent in sents] # all features seprated by '/'
            senttokens = [sent.split('/')[0] for sent in sents]
            sents = ' '.join(sents)
            senttokens = ' '.join(senttokens)
            return sents.strip(), senttokens.strip()

        xmltree = ET.parse(filename)
        root = xmltree.getroot()
        dateCreated = root.attrib[XML_ATTRIB_REFDATE] # assumed to be in default YYYY-MM-DD or the process breaks

        sentences = [] # to hold the list of sentences in the file built from everything
        sentencestokens = [] # holds sentences built from tokens only

        for sent in xmltree.iter('s'):

            sent, senttoken = make_sent(ET.tostring(sent, method='text').decode(self.decoding))

            # some html and xml unfriendly characters throw off processing..
            senttoken = html.unescape(senttoken)
            senttoken = self.replace_xml_chars(senttoken)

            sentences.append(sent)
            sentencestokens.append(senttoken)

        # now call heideltime to process the sentence
        text = '\n'.join(sentencestokens)
        # the main course...you should have had your appetizers by now..
        result = self.hw.parse(text,dateCreated)

        # We'll assume here that the list of dates returned matches the sentence indexes properly.
        # This has been empirically proven.
        dates = self.parse_timex3_xml(result)

        # Testing only
        df = pd.concat([pd.Series(sentences),pd.Series(dates)],axis=1)
        df.to_csv('test.csv')

        # TODO: parse xml, extract all time phrases - done
        #  classify as TP, then add as TEI xml - need to pickle model

    def replace_xml_chars(self,text):
        return text.replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;').replace('"', '&quot;').replace("'",'&apos;')  # assumes these dont affect time recognition..

    def get_attr(self, xml):
        """
        Helper method
        """

        attributes = []
        if len(xml.attrib) != 0:
            attributes.append(xml.attrib)
        return attributes

    def parse_timex3_xml(self, xmltext):

        """
        Parses the TimeX3 file and returns hypotheses separated by ';'
        :param xmlfile: The xml file output with TIMEX3 tags created by the tool
        :return: list of dates for each GUM sentence separated by ';'
        """

        xmltree = ET.fromstring(xmltext)

        # extract the sentences
        for node in xmltree.iter(XML_ROOT_TIMEX3):
            s = html.unescape(ET.tostring(node, method='text').decode(self.decoding, 'ignore'))

        s = s.split('\n')
        s = [t for t in s if t]  # removes extra spaces

        # fetch all the dates
        all_dates = []
        for node in xmltree.iter('TIMEX3'):
            attributes = self.get_attr(node)
            if len(attributes) > 0:
                if attributes[0]['type'] == "SET": continue # disregard sets entirely from being annotated.

            if '\n' in node.text: continue  # invalid date has been tagged
            all_dates.append(node.text)

        # Now 'assign' the date to the text sentence
        i = 0
        j = 0

        sorted_dates = []
        dates = []
        while i < len(s):
            if j == len(all_dates):
                sorted_dates.append(dates)
                break

            if all_dates[j] in s[i]:
                dates.append(all_dates[j])
                j += 1
            else:
                sorted_dates.append(dates)
                dates = []
                i += 1

        sorted_dates = [';'.join(date) for date in sorted_dates]

        return sorted_dates

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
    dtr = DateTimeRecognizer(heideltimeobj=hw)
    dtr.run(input_dir='/home/gooseg/Desktop/amalgum/amalgum/target/08_RSTParser/xml/',output_dir=None)


    pass


if __name__ == "__main__":
    # Testing only
    main()