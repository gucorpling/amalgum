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
from datetime import datetime


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
        self.featuredict = featuredict = {'obl':0,'obl:npmod':0,'obl:tmod':0,'nsubj':0,'nsubj:pass':0,'obj':0,'iobj':0,'csubj':0,'csubj:pass':0,'ccomp':0,'xcomp':0,'nummod':0,'acl':0,'amod':0,'appos':0,'acl:relcl':0,'det':0,'det:predet':0,'neg':0,'nmod':0,'case':0,'nmod:npmod':0,'nmod:tmod':0,'nmod:poss':0,'advcl':0,'advmod':0,'compound':0,'compound:prt':0,'flat':0,'fixed':0,'foreign':0,'goeswith':0,'list':0,'dislocated':0,'parataxis':0,'orphan':0,'reparandum':0,'vocative':0,'discourse':0,'expl':0,'aux':0,'aux:pass':0,'cop':0,'mark':0,'punct':0,'conj':0,'cc':0,'cc:preconj':0,'punct':0,'root':0,'dep':0,'$':0, "''":0, ',':0, '-LRB-':0, '-LSB-':0, '-RRB-':0, '-RSB-':0, '.':0 ,':':0, 'ADD':0, 'ADJ':0,'NP':0, 'ADP':0,'ADV':0, 'AFX':0, 'AUX':0, 'CC':0, 'CCONJ':0, 'CD':0, 'DET':0, 'DT':0, 'EX':0, 'FW':0, 'GW':0, 'HYPH':0, 'IN':0,'INTJ':0, 'JJ':0, 'JJR':0, 'JJS':0, 'LS':0, 'MD':0, 'NFP':0, 'NN':0, 'NNP':0, 'NNPS':0, 'NNS':0, 'NOUN':0,'NUM':0, 'PART':0, 'PDT':0, 'POS':0, 'PRON':0, 'PROPN':0, 'PRP':0, 'PRP$':0, 'PUNCT':0, 'RB':0, 'RBR':0,'RBS':0, 'RP':0, 'SYM':0, 'TO':0, 'UH':0, 'VB':0, 'VBD':0, 'VBG':0, 'VBN':0, 'VBP':0, 'VBZ':0, 'VERB':0,'WDT':0, 'WP':0, 'WP$':0, 'WRB':0, 'X':0, '``':0,'january':0,'february':0,'march':0,'april':0,'may':0,'june':0,'july':0,'august':0,'september':0,'october':0,'november':0,'december':0,'summer':0,'winter':0,'autumn':0,'spring':0,'christmas':0,'christmas_eve':0,'easter':0,'easter_sunday':0,'monday':0,'tuesday':0,'wednesday':0,'thursday':0,'friday':0,'saturday':0,'sunday':0,'start1_obl':0,'start1_obl:npmod':0,'start1_obl:tmod':0,'start1_nsubj':0,'start1_nsubj:pass':0,'start1_obj':0,'start1_iobj':0,'start1_csubj':0,'start1_csubj:pass':0,'start1_ccomp':0,'start1_xcomp':0,'start1_nummod':0,'start1_acl':0,'start1_amod':0,'start1_appos':0,'start1_acl:relcl':0,'start1_det':0,'start1_det:predet':0,'start1_neg':0,'start1_nmod':0,'start1_case':0,'start1_nmod:npmod':0,'start1_nmod:tmod':0,'start1_nmod:poss':0,'start1_advcl':0,'start1_advmod':0,'start1_compound':0,'start1_compound:prt':0,'start1_flat':0,'start1_fixed':0,'start1_foreign':0,'start1_goeswith':0,'start1_list':0,'start1_dislocated':0,'start1_parataxis':0,'start1_orphan':0,'start1_reparandum':0,'start1_vocative':0,'start1_discourse':0,'start1_expl':0,'start1_aux':0,'start1_aux:pass':0,'start1_cop':0,'start1_mark':0,'start1_punct':0,'start1_conj':0,'start1_cc':0,'start1_cc:preconj':0,'start1_punct':0,'start1_root':0,'start1_dep':0,'start1_$':0, "start1_''":0, 'start1_,':0, 'start1_-LRB-':0, 'start1_-LSB-':0, 'start1_-RRB-':0, 'start1_-RSB-':0, 'start1_.':0 ,'start1_:':0, 'start1_ADD':0, 'start1_ADJ':0,'start1_NP':0, 'start1_ADP':0,'start1_ADV':0, 'start1_AFX':0, 'start1_AUX':0, 'start1_CC':0, 'start1_CCONJ':0, 'start1_CD':0, 'start1_DET':0, 'start1_DT':0, 'start1_EX':0, 'start1_FW':0, 'start1_GW':0, 'start1_HYPH':0, 'start1_IN':0,'start1_INTJ':0, 'start1_JJ':0, 'start1_JJR':0, 'start1_JJS':0, 'start1_LS':0, 'start1_MD':0, 'start1_NFP':0, 'start1_NN':0, 'start1_NNP':0, 'start1_NNPS':0, 'start1_NNS':0, 'start1_NOUN':0,'start1_NUM':0, 'start1_PART':0, 'start1_PDT':0, 'start1_POS':0, 'start1_PRON':0, 'start1_PROPN':0, 'start1_PRP':0, 'start1_PRP$':0, 'start1_PUNCT':0, 'start1_RB':0, 'start1_RBR':0,'start1_RBS':0, 'start1_RP':0, 'start1_SYM':0, 'start1_TO':0, 'start1_UH':0, 'start1_VB':0, 'start1_VBD':0, 'start1_VBG':0, 'start1_VBN':0, 'start1_VBP':0, 'start1_VBZ':0, 'start1_VERB':0,'start1_WDT':0, 'start1_WP':0, 'start1_WP$':0, 'start1_WRB':0, 'start1_X':0, 'start1_``':0,'end1_obl':0,'end1_obl:npmod':0,'end1_obl:tmod':0,'end1_nsubj':0,'end1_nsubj:pass':0,'end1_obj':0,'end1_iobj':0,'end1_csubj':0,'end1_csubj:pass':0,'end1_ccomp':0,'end1_xcomp':0,'end1_nummod':0,'end1_acl':0,'end1_amod':0,'end1_appos':0,'end1_acl:relcl':0,'end1_det':0,'end1_det:predet':0,'end1_neg':0,'end1_nmod':0,'end1_case':0,'end1_nmod:npmod':0,'end1_nmod:tmod':0,'end1_nmod:poss':0,'end1_advcl':0,'end1_advmod':0,'end1_compound':0,'end1_compound:prt':0,'end1_flat':0,'end1_fixed':0,'end1_foreign':0,'end1_goeswith':0,'end1_list':0,'end1_dislocated':0,'end1_parataxis':0,'end1_orphan':0,'end1_reparandum':0,'end1_vocative':0,'end1_discourse':0,'end1_expl':0,'end1_aux':0,'end1_aux:pass':0,'end1_cop':0,'end1_mark':0,'end1_punct':0,'end1_conj':0,'end1_cc':0,'end1_cc:preconj':0,'end1_punct':0,'end1_root':0,'end1_dep':0,'end1_$':0, "end1_''":0, 'end1_,':0, 'end1_-LRB-':0, 'end1_-LSB-':0, 'end1_-RRB-':0, 'end1_-RSB-':0, 'end1_.':0 ,'end1_:':0, 'end1_ADD':0, 'end1_ADJ':0,'end1_NP':0, 'end1_ADP':0,'end1_ADV':0, 'end1_AFX':0, 'end1_AUX':0, 'end1_CC':0, 'end1_CCONJ':0, 'end1_CD':0, 'end1_DET':0, 'end1_DT':0, 'end1_EX':0, 'end1_FW':0, 'end1_GW':0, 'end1_HYPH':0, 'end1_IN':0,'end1_INTJ':0, 'end1_JJ':0, 'end1_JJR':0, 'end1_JJS':0, 'end1_LS':0, 'end1_MD':0, 'end1_NFP':0, 'end1_NN':0, 'end1_NNP':0, 'end1_NNPS':0, 'end1_NNS':0, 'end1_NOUN':0,'end1_NUM':0, 'end1_PART':0, 'end1_PDT':0, 'end1_POS':0, 'end1_PRON':0, 'end1_PROPN':0, 'end1_PRP':0, 'end1_PRP$':0, 'end1_PUNCT':0, 'end1_RB':0, 'end1_RBR':0,'end1_RBS':0, 'end1_RP':0, 'end1_SYM':0, 'end1_TO':0, 'end1_UH':0, 'end1_VB':0, 'end1_VBD':0, 'end1_VBG':0, 'end1_VBN':0, 'end1_VBP':0, 'end1_VBZ':0, 'end1_VERB':0,'end1_WDT':0, 'end1_WP':0, 'end1_WP$':0, 'end1_WRB':0, 'end1_X':0, 'end1_``':0,'end2_obl':0,'end2_obl:npmod':0,'end2_obl:tmod':0,'end2_nsubj':0,'end2_nsubj:pass':0,'end2_obj':0,'end2_iobj':0,'end2_csubj':0,'end2_csubj:pass':0,'end2_ccomp':0,'end2_xcomp':0,'end2_nummod':0,'end2_acl':0,'end2_amod':0,'end2_appos':0,'end2_acl:relcl':0,'end2_det':0,'end2_det:predet':0,'end2_neg':0,'end2_nmod':0,'end2_case':0,'end2_nmod:npmod':0,'end2_nmod:tmod':0,'end2_nmod:poss':0,'end2_advcl':0,'end2_advmod':0,'end2_compound':0,'end2_compound:prt':0,'end2_flat':0,'end2_fixed':0,'end2_foreign':0,'end2_goeswith':0,'end2_list':0,'end2_dislocated':0,'end2_parataxis':0,'end2_orphan':0,'end2_reparandum':0,'end2_vocative':0,'end2_discourse':0,'end2_expl':0,'end2_aux':0,'end2_aux:pass':0,'end2_cop':0,'end2_mark':0,'end2_punct':0,'end2_conj':0,'end2_cc':0,'end2_cc:preconj':0,'end2_punct':0,'end2_root':0,'end2_dep':0,'end2_$':0, "end2_''":0, 'end2_,':0, 'end2_-LRB-':0, 'end2_-LSB-':0, 'end2_-RRB-':0, 'end2_-RSB-':0, 'end2_.':0 ,'end2_:':0, 'end2_ADD':0, 'end2_ADJ':0,'end2_NP':0, 'end2_ADP':0,'end2_ADV':0, 'end2_AFX':0, 'end2_AUX':0, 'end2_CC':0, 'end2_CCONJ':0, 'end2_CD':0, 'end2_DET':0, 'end2_DT':0, 'end2_EX':0, 'end2_FW':0, 'end2_GW':0, 'end2_HYPH':0, 'end2_IN':0,'end2_INTJ':0, 'end2_JJ':0, 'end2_JJR':0, 'end2_JJS':0, 'end2_LS':0, 'end2_MD':0, 'end2_NFP':0, 'end2_NN':0, 'end2_NNP':0, 'end2_NNPS':0, 'end2_NNS':0, 'end2_NOUN':0,'end2_NUM':0, 'end2_PART':0, 'end2_PDT':0, 'end2_POS':0, 'end2_PRON':0, 'end2_PROPN':0, 'end2_PRP':0,'end2_PP$':0,'end1_PP$':0,'start2_PP$':0,'start1_PP$':0,'end2_PP':0,'end1_PP':0,'start2_PP':0,'start1_PP':0, 'end2_PRP$':0, 'end2_PUNCT':0, 'end2_RB':0, 'end2_RBR':0,'end2_RBS':0, 'end2_RP':0, 'end2_SYM':0, 'end2_TO':0, 'end2_UH':0, 'end2_VB':0, 'end2_VBD':0, 'end2_VBG':0, 'end2_VBN':0, 'end2_VBP':0, 'end2_VBZ':0, 'end2_VERB':0,'end2_WDT':0, 'end2_WP':0, 'end2_WP$':0, 'end2_WRB':0, 'end2_X':0, 'end2_``':0,'start2_obl':0,'start2_obl:npmod':0,'start2_obl:tmod':0,'start2_nsubj':0,'start2_nsubj:pass':0,'start2_obj':0,'start2_iobj':0,'start2_csubj':0,'start2_csubj:pass':0,'start2_ccomp':0,'start2_xcomp':0,'start2_nummod':0,'start2_acl':0,'start2_amod':0,'start2_appos':0,'start2_acl:relcl':0,'start2_det':0,'start2_det:predet':0,'start2_neg':0,'start2_nmod':0,'start2_case':0,'start2_nmod:npmod':0,'start2_nmod:tmod':0,'start2_nmod:poss':0,'start2_advcl':0,'start2_advmod':0,'start2_compound':0,'start2_compound:prt':0,'start2_flat':0,'start2_fixed':0,'start2_foreign':0,'start2_goeswith':0,'start2_list':0,'start2_dislocated':0,'start2_parataxis':0,'start2_orphan':0,'start2_reparandum':0,'start2_vocative':0,'start2_discourse':0,'start2_expl':0,'start2_aux':0,'start2_aux:pass':0,'start2_cop':0,'start2_mark':0,'start2_punct':0,'start2_conj':0,'start2_cc':0,'start2_cc:preconj':0,'start2_punct':0,'start2_root':0,'start2_dep':0,'start2_$':0, "start2_''":0, 'start2_,':0, 'start2_-LRB-':0, 'start2_-LSB-':0, 'start2_-RRB-':0, 'start2_-RSB-':0, 'start2_.':0 ,'start2_:':0, 'start2_ADD':0, 'start2_ADJ':0,'start2_NP':0, 'start2_ADP':0,'start2_ADV':0, 'start2_AFX':0, 'start2_AUX':0, 'start2_CC':0, 'start2_CCONJ':0, 'start2_CD':0, 'start2_DET':0, 'start2_DT':0, 'start2_EX':0, 'start2_FW':0, 'start2_GW':0, 'start2_HYPH':0, 'start2_IN':0,'start2_INTJ':0, 'start2_JJ':0, 'start2_JJR':0, 'start2_JJS':0, 'start2_LS':0, 'start2_MD':0, 'start2_NFP':0, 'start2_NN':0, 'start2_NNP':0, 'start2_NNPS':0, 'start2_NNS':0, 'start2_NOUN':0,'start2_NUM':0, 'start2_PART':0, 'start2_PDT':0, 'start2_POS':0, 'start2_PRON':0, 'start2_PROPN':0, 'start2_PRP':0, 'start2_PRP$':0, 'start2_PUNCT':0, 'start2_RB':0, 'start2_RBR':0,'start2_RBS':0, 'start2_RP':0, 'start2_SYM':0, 'start2_TO':0, 'start2_UH':0, 'start2_VB':0, 'start2_VBD':0, 'start2_VBG':0, 'start2_VBN':0, 'start2_VBP':0, 'start2_VBZ':0, 'start2_VERB':0,'start2_WDT':0, 'start2_WP':0, 'start2_WP$':0, 'start2_WRB':0, 'start2_X':0, 'start2_``':0,'day':0,'century':0,'millenia':0,'hour':0,'minute':0,'year':0,'second':0,'month':0,'start2_day':0,'start2_century':0,'start2_millenia':0,'start2_hour':0,'start2_minute':0,'start2_year':0,'start2_second':0,'start2_month':0,'start2_day':0,'start2_century':0,'start2_millenia':0,'start2_hour':0,'start2_minute':0,'start2_year':0,'start2_second':0,'start2_month':0,'start1_day':0,'start1_century':0,'start1_millenia':0,'start1_hour':0,'start1_minute':0,'start1_year':0,'start1_second':0,'start1_month':0,'end2_day':0,'end2_century':0,'end2_millenia':0,'end2_hour':0,'end2_minute':0,'end2_year':0,'end2_second':0,'end2_month':0,'end1_day':0,'end1_century':0,'end1_millenia':0,'end1_hour':0,'end1_minute':0,'end1_year':0,'end1_second':0,'end1_month':0,'start1_january':0,'start1_february':0,'start1_march':0,'start1_april':0,'start1_may':0,'start1_june':0,'start1_july':0,'start1_august':0,'start1_september':0,'start1_october':0,'start1_november':0,
               'start1_december':0,'start1_summer':0,'start1_winter':0,'start1_autumn':0,'start1_spring':0,'start1_christmas':0,'start1_christmas_eve':0,'start1_easter':0,'start1_easter_sunday':0,'start1_monday':0,'start1_tuesday':0,'start1_wednesday':0,'start1_thursday':0,'start1_friday':0,'start1_saturday':0,'start1_sunday':0,'start2_january':0,'start2_february':0,'start2_march':0,'start2_april':0,'start2_may':0,'start2_june':0,'start2_july':0,'start2_august':0,'start2_september':0,'start2_october':0,'start2_november':0,'start2_december':0,'start2_summer':0,'start2_winter':0,'start2_autumn':0,'start2_spring':0,'start2_christmas':0,'start2_christmas_eve':0,'start2_easter':0,'start2_easter_sunday':0,'start2_monday':0,'start2_tuesday':0,'start2_wednesday':0,'start2_thursday':0,'start2_friday':0,'start2_saturday':0,'start2_sunday':0,'end2_january':0,'end2_february':0,'end2_march':0,'end2_april':0,'end2_may':0,'end2_june':0,'end2_july':0,'end2_august':0,'end2_september':0,'end2_october':0,'end2_november':0,'end2_december':0,'end2_summer':0,'end2_winter':0,'end2_autumn':0,'end2_spring':0,'end2_christmas':0,'end2_christmas_eve':0,'end2_easter':0,'end2_easter_sunday':0,'end2_monday':0,'end2_tuesday':0,'end2_wednesday':0,'end2_thursday':0,'end2_friday':0,'end2_saturday':0,'end2_sunday':0,'end1_january':0,'end1_february':0,'end1_march':0,'end1_april':0,'end1_may':0,'end1_june':0,'end1_july':0,'end1_august':0,'end1_september':0,'end1_october':0,'end1_november':0,'end1_december':0,'end1_summer':0,'end1_winter':0,'end1_autumn':0,'end1_spring':0,'end1_christmas':0,'end1_christmas_eve':0,'end1_easter':0,'end1_easter_sunday':0,'end1_monday':0,'end1_tuesday':0,'end1_wednesday':0,'end1_thursday':0,'end1_friday':0,'end1_saturday':0,'end1_sunday':0}

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

        endindex = startindex + len(phrase.split()) - 1

        # start building the features now that we have the indices of the phrase in the sentence
        # Features are basically count vectors of the defined featureset in the template
        fdict = self.featuredict.copy() # need to do deep copy from the template
        featurelist = fdict.keys()

        # This will check features that are specific word tokens in the sentence, e.g January, Tuesday,
        for i in range(startindex,endindex + 1,1):
            feats = features[i].split('/') # get the token word
            if str(feats[1]).lower() in featurelist:
                fdict[str(feats[1]).lower()] += 1 # Increment if the word is in the feature list
            if str(feats[4]) in fdict.keys():
                fdict[str(feats[4])] += 1 # the Penn treebank POS tag
            if str(feats[7]) in fdict.keys():
                fdict[str(feats[7])] += 1 # the UD tag


        if startindex > 0:
            feats = features[startindex - 1].split('/')  # get the token word
            if str(feats[1]).lower() in fdict.keys():
                fdict['start1_' + str(feats[1]).lower()] += 1
            if str(feats[4]) in fdict.keys():
                fdict['start1_' + str(feats[4])] += 1
            if str(feats[7]) in fdict.keys():
                fdict['start1_' + str(feats[7])] += 1

        if startindex > 1:
            feats = features[startindex - 2].split('/')  # get the token word
            if str(feats[1]).lower() in fdict.keys():
                fdict['start2_' + str(feats[1]).lower()] += 1
            if str(feats[4]) in fdict.keys():
                fdict['start2_' + str(feats[4])] += 1
            if str(feats[7]) in fdict.keys():
                fdict['start2_' + str(feats[7])] += 1

        if endindex + 1 < len(features):
            feats = features[endindex + 1].split('/')  # get the token word
            if str(feats[1]).lower() in fdict.keys():
                fdict['end1_' + str(feats[1]).lower()] += 1
            if str(feats[4]) in fdict.keys():
                fdict['end1_' + str(feats[4])] += 1
            if str(feats[7]) in fdict.keys():
                fdict['end1_' + str(feats[7])] += 1

        if endindex + 2 < len(features):
            feats = features[endindex + 2].split('/')  # get the token word
            if str(feats[1]).lower() in fdict.keys():
                fdict['end2_' + str(feats[1]).lower()] += 1
            if str(feats[4]) in fdict.keys():
                fdict['end2_' + str(feats[4])] += 1
            if str(feats[7]) in fdict.keys():
                fdict['end2_' + str(feats[7])] += 1


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
        fdict['NN_obl'] = fdict['NN'] * fdict['obl']
        fdict['NP_nummod'] = fdict['NP'] * fdict['nummod']
        fdict['NN_nummod'] = fdict['NN'] * fdict['nummod']
        fdict['NP_nmod'] = fdict['NP'] * fdict['nmod']
        fdict['NN_nmod'] = fdict['NN'] * fdict['nmod']
        fdict['NP_compound'] = fdict['NP'] * fdict['compound']
        fdict['NN_compound'] = fdict['NN'] * fdict['compound']
        fdict['RB_amod'] = fdict['RB'] * fdict['amod']
        fdict['JJ_amod'] = fdict['JJ'] * fdict['amod']
        fdict['NN_obltmod'] = fdict['NN'] * fdict['obl:tmod']

        # these become useful when adding the tags to the xml
        fdict['sentence_index'] = index + 1
        fdict['start_index'] = int(startindex)
        fdict['phrase'] = phrase
        fdict['timextype'] = timextype
        fdict['timexvalue'] = timexvalue


        return fdict


class DateTimeRecognizer(NLPModule):
    def __init__(self,heideltimeobj,datefilterobj,postaglabelencoderobj,decoding='ascii'):

        super().__init__(config=None)
        self.decoding = decoding
        self.hw = heideltimeobj
        self.datefilter = datefilterobj

        self.regexnonchars = r'[^0-9-]' # anything not a number or hyphen
        self.regexyyyymmdd = r'[0-9]{4}-[0-9]{2}-[0-9]{2}' # matches YYYY-MM-DD
        self.regexmmdd = r'--[0-9]{2}-[0-9]{2}' #--mm-dd, unlikely to be found in TimeML
        self.regexmm = r'--[0-9]{2}'  # matches --mm, unlikely
        self.regexyyyymm= r'[0-9]{4}-[0-9]{2}'  # matches YYYY-MM, likely!
        self.regexyyyy = r'[0-9]{4}'  # matches YYYY, likely!
        self.seasons = {'SU':['from:--06','to:--09'],'WI':['from:--12','to:--03'],'FA':['from:--09','to:--12'],'SP':['from:--03','to:--06']}
        self.holidates = {'spanish golden age':['from:1556','to:1659'],'easter':['notBefore:--03','notAfter:--05'],'easter sunday':['notBefore:--03','notAfter:--05'],'christmas':['when:--12-25'],'christmas eve':['when:--12-24'],'world war 2':['from:1939-09-01','to:1945-02-01'],'world war ii':['from:1939-09-01','to:1945-02-01'],'world war 1':['from:1914','to:1918'],'world war i':['from:1914','to:1918'],'the american revolution':['notBefore:1775','notAfter:1783'],'the american revolutionary war':['notBefore:1775','notAfter:1783'],'the civil war':['notBefore:1861','notAfter:1865'],'the american civil war':['notBefore:1861','notAfter:1865'],'the reconstruction era':['notBefore:1863','notAfter:1887']}

        # TODO: 'soft-wire' feature names from the label encoder for pos tagging instead of hard-wiring in the dictionary features
        #self.cd = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        #with open(self.cd + postaglabelencoderobj, "rb") as f:
        #    le = pickle.load(f)

    def requires(self):
        pass

    def provides(self):
        pass

    def timex_to_tei(self,timextype,timexvalue):
        """
        Converts TIMEX3 values to TEI encodings
        Returns the tag type with attributes to build the date/time tag
        """

        # TODO:
        # check min date, sometimes 200 years apart!
        # gazette for listed holidayes
        # from-to or notbefore-notafter
        # centuries, decades
        # times

        def striphyphens(text):
            if text[-1:] == '-': text = text[:-1]
            if text[:1] == '-' and text[:2] != '-': text = text[1:]
            return text

        # init
        result = {}

        temp = re.sub(self.regexnonchars, '', timexvalue)
        temp = striphyphens(temp)

        # see if yyyy-mm-dd. -mm-dd, or --mm matched
        if re.match(self.regexyyyymmdd, temp) or re.match(self.regexmmdd, temp) \
                or re.match(self.regexmm,temp) or re.match(self.regexyyyymm, temp):
            result['date'] = ['when:' + temp]
            return result

        for item in self.seasons.keys():
            if item in timexvalue:
                if re.match(self.regexyyyy,temp):
                    result['date'] = [i.replace('--',temp + '-') for i in self.seasons[item]]
                else:
                    result['date'] = self.seasons[item]
                break

        return result


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
            Recursively builds the new xml file in memory in place, and stamps the date xml on it
            don't remove the counter, it is an accumulator that keep tracks of how many sentences we have iterated over
            """

            #TODO: deconstruct and reconstruct all elements in a sentence along with the dates
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

                            attributes = self.timex_to_tei(str(row['timextype']), str(row['timexvalue']))
                            for key, value in attributes.items():  # just one attribute in the collection

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

                                date = ET.Element(key)
                                for attribs in value :
                                    date.set(attribs.split(':')[0],attribs.split(':')[1])

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

        xmltree = ET.parse(filename)
        root = xmltree.getroot()

        if XML_ATTRIB_REFDATE  in root.attrib:
            dateCreated = root.attrib[XML_ATTRIB_REFDATE] # assumed to be in default YYYY-MM-DD or the process breaks
        else:
            dateCreated = datetime.today().strftime('%Y-%m-%d')

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

        if len(inferencedf) != 0:
            indexphrases = inferencedf[['sentence_index','start_index','phrase','timextype','timexvalue']]
            inferencedf.drop(columns=['sentence_index','start_index','phrase','timextype','timexvalue'],axis=1,inplace=True)

            # Filter the dates that dont pass GUM annotated standards..
            tpprobs = self.datefilter.rf.predict(inferencedf)

            indexphrases['label'] = pd.Series(tpprobs)
            #indexphrases = indexphrases.loc[indexphrases['label'] == 1]
            print(indexphrases)

            if indexphrases is not None and len(indexphrases) != 0: # only if we have dates..
                indexphrases.sort_values(['sentence_index','start_index'],ascending=[True,False],inplace=True)

                # just in case..this leads to selection of only 1 date element in the same token....
                indexphrases= indexphrases.groupby(by=['sentence_index','start_index']).head(1)
                sentenceindices = set(indexphrases['sentence_index'].tolist())

                # Build the xml with the new date tag
                _ = add_datetime_tags(root) # modify xml in place and add date tags

        # write to disk
        tree = ET.ElementTree(root)
        return tree

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
            file = '/home/gooseg/Desktop/amalgum/amalgum/target/04_DepParser/xml/autogum_bio_doc005.xml'
            print(file)
            treeobj = self.process_file(file)
            treeobj.write(open(output_dir + file.split('/')[-1], 'w'), encoding='unicode', xml_declaration=True)
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
    dtr.run(input_dir='/home/gooseg/Desktop/amalgum/amalgum/target/04_DepParser/xml/',output_dir='/home/gooseg/Desktop/amalgum/amalgum/target/testdate/')
    print (time.time() - start)



    pass


if __name__ == "__main__":
    # Testing only
    main()