## feature.py
## Author: Yangfeng Ji
## Date: 08-29-2014
## Time-stamp: <yangfeng 09/24/2015 16:21:12>

from .util import getgrams, getbc
import gzip, sys

PY3 = sys.version_info[0] > 2

if PY3:
    from pickle import load
else:
    from cPickle import load


class FeatureGenerator(object):
    def __init__(self, stack, queue, doc, bcvocab, nprefix=10):
        """ Initialization of feature generator

        Currently, we only consider the feature generated
        from the top 2 spans from the stack, and the first
        span from the queue. However, you are available to
        use any other information for feature generation.
        - YJ
        
        :type stack: list
        :param stack: list of Node instance

        :type queue: list
        :param queue: list of Node instance

        :type doc: Doc instance
        :param doc: 
        """
        # Predefined variables
        self.npref = nprefix
        # Load Brown clusters
        self.bcvocab = bcvocab
        # -------------------------------------
        self.doc = doc
        # Stack
        if len(stack) >= 2:
            self.top1span, self.top2span = stack[-1], stack[-2]
        elif len(stack) == 1:
            self.top1span, self.top2span = stack[-1], None
        else:
            self.top1span, self.top2span = None, None
        # Queue
        if len(queue) > 0:
            self.firstspan = queue[0]
        else:
            self.firstspan = None
        # Doc length wrt EDUs
        self.doclen = len(self.doc.edudict)
        #self.genre = self.doc.tokendict[0].ner
        self.genre = self.doc.tokendict[0].genre



    def features(self, new=False):
        """ Main function to generate features
        """
        featlist = []
        ## Status features (Basic features)
        for feat in self.status_features():
            featlist.append(feat)
        ## Lexical features
        for feat in self.lexical_features():
            featlist.append(feat)
        ## Structural features
        for feat in self.structural_features():
            featlist.append(feat)
        ## EDU features
        if new:
            for feat in self.edu_features_new():
                featlist.append(feat)
        else:
            for feat in self.edu_features():
                featlist.append(feat)
        ## Distributional representation
        for feat in self.distributional_features():
            featlist.append(feat)
        for feat in self.nucleus_features():
            featlist.append(feat)
        ## Brown clusters
        if self.bcvocab is not None:
            if new:
                for feat in self.bc_features_new():
                    featlist.append(feat)
            else:
                for feat in self.bc_features():
                    featlist.append(feat)
        ## Rich features
        for feat in self.rich_features():
           featlist.append(feat)

        return featlist

    def structural_features(self):
        """ Structural features

        TODO: add a upper/lower thresholds
        """
        features = []
        if self.top1span is not None:
            span = self.top1span
            # Span Length wrt EDUs
            edulen1 = span.eduspan[1]-span.eduspan[0]+1
            yield ('Top1-Stack','Length-EDU', edulen1)
            # Distance to the beginning of the document wrt EDUs
            yield ('Top1-Stack','Dist-To-Begin',span.eduspan[0])
            # Distance to the end of the document wrt EDUs
            yield ('Top1-Stack','Dist-To-End',self.doclen-span.eduspan[1])
        if self.top2span is not None:
            span = self.top2span
            edulen2 = span.eduspan[1]-span.eduspan[0]+1
            yield ('Top2-Stack','Length-EDU', edulen2)
            yield ('Top2-Stack','Dist-To-Begin',span.eduspan[0])
            yield ('Top2-Stack','Dist-To-End',self.doclen-span.eduspan[1])
        # if (self.top1span is not None) and (self.top2span is not None):
        #     if edulen1 > edulen2:
        #         yield ('Top-Stack','EDU-Comparison',True)
        #     elif edulen1 < edulen2:
        #         yield ('Top-Stack','EDU-Comparison',False)
        #     else:
        #         yield ('Top-Stack','EDU-Comparison','Equal')
        if self.firstspan is not None:
            span = self.firstspan
            yield ('First-Queue','Dist-To-Begin',span.eduspan[0])
        

    def status_features(self):
        """ Features related to stack/queue status
        """
        # Stack
        if (self.top1span is None) and (self.top2span is None):
            yield ('Stack','Empty')
        elif (self.top1span is not None) and (self.top2span is None):
            yield ('Stack', 'OneElem')
        elif (self.top1span is not None) and (self.top2span is not None):
            yield ('Stack', 'MoreElem')
        else:
            raise ValueError("Unrecognized stack status")
        # Queue
        if (self.firstspan is None):
            yield ('Queue', 'Empty')
        else:
            yield ('Queue', 'NonEmpty')


    def edu_features(self):
        """ Features about EDUs in one text span
        """
        # ---------------------------------------
        # ---------------------------------------
        # Whether within same sentence
        # Span 1 and 2
        # Last word from span 1, first word from span 2
        try:
            text1, text2 = self.top1span.text, self.top2span.text
            if (self.doc.tokendict[text1[-1]].sidx == self.doc.tokendict[text2[0]].sidx):
                yield ('Top12-Stack', 'SameSent', True)
            else:
                yield ('Top12-Stack', 'SameSent', False)
        except AttributeError:
            yield ('Top12-Stack', 'SameSent', None)
        # Span 1 and top span
        # First word from span 1, last word from span 3
        try:
            text1, text3 = self.top1span.text, self.firstspan.text
            if (self.doc.tokendict[text1[0]].sidx == self.doc.tokendict[text3[-1]].sidx):
                yield ('Stack-Queue', 'SameSent', True)
            else:
                yield ('Stack-Queue', 'SameSent', False)
        except AttributeError:
            yield ('Stack-Queue', 'SameSent', None)


    def edu_features_new(self):
        """ Features about EDUs in one text span
        """
        def edu_transition_feat(self,feat_name,attr_name,ftype="bool",comparison="Top12-Stack",location=None,float_thresh=0.5):
            try:
                if comparison == "Top12-Stack":
                    # Span 1 and 2
                    # Last word from span 1, first word from span 2: location=="transition"
                    text1, text2 = self.top1span.text, self.top2span.text
                elif comparison == "Stack-Queue":
                    # Span 1 and top span
                    # First word from span 1, last word from span 3: location=="edges"
                    text2, text1 = self.top1span.text, self.firstspan.text  # text2 == old text3
                else:
                    raise ValueError("No such comparison type: " + comparison)
                sidx1 = self.doc.tokendict[text1[-1]].sidx
                sidx2 = self.doc.tokendict[text2[0]].sidx
                if location is None:
                    pivot_token = self.doc.tokendict[text1[-1]]
                    aux_token = self.doc.tokendict[text2[0]]
                elif location == "initial":  # Take information from first tokens of each span
                    pivot_token = self.doc.tokendict[text1[0]]
                    aux_token = self.doc.tokendict[text2[0]]
                else:
                    raise ValueError("No such location type: " + location)
                if ftype == "bool":
                    if sidx1 == sidx2:  # Same sentence
                        return (comparison, feat_name, True)
                    else:
                        return (comparison, feat_name, False)
                elif ftype == "cat":
                    if sidx1 == sidx2:
                        return (comparison, feat_name, pivot_token.__dict__[attr_name])
                    else:
                        return (comparison, feat_name, pivot_token.__dict__[attr_name] + "_" + aux_token.__dict__[attr_name])
                elif ftype == "float":
                    num_val = aux_token.__dict__[attr_name]
                    if 0 <= num_val < 0.25:
                        val = "SmallPos"
                    elif float_thresh <= num_val:
                        val= "BigPos"
                    elif -1*float_thresh <= num_val < 0:
                        val = "SmallNeg"
                    else:
                        val = "BigNeg"
                    return (comparison, feat_name, val)

            except AttributeError:
                return (comparison, feat_name, None)


        # ---------------------------------------
        # EDU length
        if self.top1span is not None:
            eduspan = self.top1span.eduspan
            #yield ('Top1-Stack', 'nEDUs', eduspan[1]-eduspan[0]+1)
        if self.top2span is not None:
            eduspan = self.top2span.eduspan
            #yield ('Top2-Stack', 'nEDUs', eduspan[1]-eduspan[0]+1)
        # ---------------------------------------
        # Whether within same sentence
        yield edu_transition_feat(self, 'SameSent', 'sidx', ftype="bool", comparison="Top12-Stack")
        yield edu_transition_feat(self, 'SameSent', 'sidx', ftype="bool", comparison="Stack-Queue")

        # s_type at transition/containing sentence
        yield edu_transition_feat(self, 'SType', 's_type', ftype="cat", comparison="Top12-Stack")
        yield edu_transition_feat(self, 'SType', 's_type', ftype="cat", comparison="Stack-Queue")

        # xml at transition/containing sentence
        yield edu_transition_feat(self, 'XML','xml',ftype="cat", comparison="Top12-Stack", location="initial")
        yield edu_transition_feat(self, 'XML','xml',ftype="cat", comparison="Stack-Queue", location="initial")

        # sentiment
        #yield edu_transition_feat(self, 'Sentiment','sentiment',ftype="float", comparison="Stack-Queue", location="initial",float_thresh=0.25)
        yield edu_transition_feat(self, 'Subjectivity','subjectivity', comparison="Top12-Stack", ftype="float", location="initial",float_thresh=0.5)
        yield edu_transition_feat(self, 'Subjectivity','subjectivity',ftype="float", comparison="Stack-Queue", location="initial",float_thresh=0.5)

        # parent_dir
        pdir = False
        if pdir:
            try:
                text1, text2 = self.top1span.text, self.top2span.text
                dir1 = "NONE"
                if any([self.doc.tokendict[k].ner=="LEFT" for k in text1]):
                    dir1 = "LEFT"
                elif any([self.doc.tokendict[k].ner=="RIGHT" for k in text1]):
                    dir1 = "RIGHT"
                dir2 = "NONE"
                if any([self.doc.tokendict[k].ner=="LEFT" for k in text2]):
                    dir2 = "LEFT"
                elif any([self.doc.tokendict[k].ner=="RIGHT" for k in text2]):
                    dir2 = "RIGHT"

                if (self.doc.tokendict[text1[-1]].sidx == self.doc.tokendict[text2[0]].sidx):
                    yield ('Top12-Stack', 'Dir', dir1)
                else:
                    yield ('Top12-Stack', 'Dir', dir1 + "_" + dir2)
            except AttributeError:
                yield ('Top12-Stack', 'Dir', None)
            # Span 1 and top span
            # First word from span 1, last word from span 3
            try:
                text1, text3 = self.top1span.text, self.firstspan.text
                dir1 = "NONE"
                if any([self.doc.tokendict[k].ner=="LEFT" for k in text1]):
                    dir1 = "LEFT"
                elif any([self.doc.tokendict[k].ner=="RIGHT" for k in text1]):
                    dir1 = "RIGHT"
                dir3 = "NONE"
                if any([self.doc.tokendict[k].ner=="LEFT" for k in text3]):
                    dir3 = "LEFT"
                elif any([self.doc.tokendict[k].ner=="RIGHT" for k in text3]):
                    dir3 = "RIGHT"

                if (self.doc.tokendict[text1[0]].sidx == self.doc.tokendict[text3[-1]].sidx):
                    yield ('Stack-Queue', 'Dir', dir1)
                else:
                    yield ('Stack-Queue', 'Dir', dir1 + "_" + dir3)
            except AttributeError:
                yield ('Stack-Queue', 'Dir', None)


    def lexical_features(self):
        """ Features about tokens in one text span
        """ 
        if self.top1span is not None:
            span = self.top1span
            # yield ('Top1-Stack', 'nTokens', len(span.text))
            grams = getgrams(span.text, self.doc.tokendict)
            for gram in grams:
                yield ('Top1-Stack', 'nGram', gram)
        if self.top2span is not None:
            span = self.top2span
            # yield ('Top2-Stack', 'nTokens', len(span.text))
            grams = getgrams(span.text, self.doc.tokendict)
            for gram in grams:
                yield ('Top2-Stack', 'nGram', gram)
        if self.firstspan is not None:
            span = self.firstspan
            # yield ('First-Queue', 'nTokens', len(span.text))
            grams = getgrams(span.text, self.doc.tokendict)
            for gram in grams:
                yield ('First-Queue', 'nGram', gram)

    def distributional_features(self):
        """ Distributional representation features proposed in
            (Ji and Eisenstein, 2014)
        """
        tokendict = self.doc.tokendict
        if self.top1span is not None:
            eduidx = self.top1span.nucedu
            for gidx in self.doc.edudict[eduidx]:
                word = tokendict[gidx].word.lower()
                yield ('DisRep', 'Top1Span', word)
        if self.top2span is not None:
            eduidx = self.top2span.nucedu
            for gidx in self.doc.edudict[eduidx]:
                word = tokendict[gidx].word.lower()
                yield ('DisRep', 'Top2Span', word)
        if self.firstspan is not None:
            eduidx = self.firstspan.nucedu
            for gidx in self.doc.edudict[eduidx]:
                word = tokendict[gidx].word.lower()
                yield ('DisRep', 'FirstSpan', word)


    def nucleus_features(self):
        """ Feature extract from one single nucleus EDU
        """
        if self.top1span is not None:
            eduidx = self.top1span.nucedu
            some_tok = self.doc.tokendict[self.doc.edudict[eduidx][0]]
            yield ('SeqPred', 'Top1Span', some_tok.seqlab)
            first_pos = self.doc.tokendict[self.doc.edudict[eduidx][0]].pos
            yield ('FirstPos', 'Top1Span', first_pos)

        if self.firstspan is not None:
            eduidx = self.firstspan.nucedu
            some_tok = self.doc.tokendict[self.doc.edudict[eduidx][0]]
            yield ('SeqPred', 'FirstSpan', some_tok.seqlab)
            first_pos = self.doc.tokendict[self.doc.edudict[eduidx][0]].pos
            yield ('FirstPos', 'FirstSpan', first_pos)

    def bc_features(self):
        """ Feature extract from brown clusters
            Features are only extracted from Nucleus EDU !!!!
        """
        tokendict = self.doc.tokendict
        edudict = self.doc.edudict
        if self.top1span is not None:
            eduidx = self.top1span.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict,
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'Top1Span', feat)
        if self.top2span is not None:
            eduidx = self.top2span.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict,
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'Top2Span', feat)
        if self.firstspan is not None:
            eduidx = self.firstspan.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict,
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'FirstSpan', feat)

    def bc_features_new(self):
        """ Feature extract from brown clusters
            Features are only extracted from Nucleus EDU !!!!
        """
        def get_position(pos,total):
            """Encode special positions in the EDU"""
            if pos == 0:
                return "First"
            elif pos == 1 and pos != total - 1 and False:
                return "Second"
            elif pos == total - 1 and False:
                return "Last"
            else:
                return ""

        tokendict = self.doc.tokendict
        edudict = self.doc.edudict
        if self.top1span is not None:
            eduidx = self.top1span.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict, 
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                position, feat = feat
                position = get_position(position,len(bcfeatures))
                yield ('BC', 'Top1Span', feat)
                if position == "First":
                    yield ('BC', 'Top1Span'+position, feat)
        if self.top2span is not None:
            eduidx = self.top2span.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict, 
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                position, feat = feat
                position = get_position(position,len(bcfeatures))
                yield ('BC', 'Top2Span', feat)
                if position == "First":
                    yield ('BC', 'Top2Span'+position, feat)
        if self.firstspan is not None:
            eduidx = self.firstspan.nucedu
            bcfeatures = getbc(eduidx, edudict, tokendict,
                               self.bcvocab, self.npref)
            for feat in bcfeatures:
                position, feat = feat
                position = get_position(position,len(bcfeatures))
                yield ('BC', 'FirstSpan', feat)
                if position == "First":
                    yield ('BC', 'FirstSpan'+position, feat)

    def rich_features(self):
        # Access additional columns
        yield ('rich','genre',self.genre)

    def make_text_with_boundaries(self, span, boundary="#"):
        last_edu = 0
        text = []
        for i in span.text:
            tok = self.doc.tokendict[i]
            if tok.eduidx != last_edu and last_edu != 0:
                text.append("#")
            text.append(tok.word)
            last_edu = tok.eduidx
        if boundary is not None and len(boundary) > 0:
            text = [boundary] + text + [boundary]
        return " ".join(text)
