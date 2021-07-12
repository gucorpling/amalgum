from .datastructure import Token, Doc
from os.path import isfile
import io


class DocReader(object):
    """ Build one doc instance from *.merge file
    """
    def __init__(self):
        """
        """
        self.fmerge = None

    def read(self, fmerge):
        """ Read information from the merge file, and create
            an Doc instance

        :type fmerge: string
        :param fmerge: merge file name
        """
        if len(fmerge) > 100 and not fmerge.endswith(".merge"):
            fin = fmerge.strip().split('\n')
        elif not isfile(fmerge):
            fin = fmerge.strip().split('\n')
        else:
            fin = io.open(fmerge, 'r', encoding='utf-8')
            # raise IOError("File doesn't exist: {}".format(fmerge))
        self.fmerge = fmerge
        gidx, tokendict = 0, {}
        # with io.open(fmerge, 'r', encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            tok = self._parseline(line)
            tokendict[gidx] = tok
            gidx += 1
        # Get EDUs from tokendict
        edudict = self._recoveredus(tokendict)
        doc = Doc()
        doc.tokendict = tokendict
        doc.edudict = edudict
        return doc

    def _parseline(self, line):
        """ Parse one line from *.merge file
        """
        items = line.split("\t")
        tok = Token()
        tok.sidx, tok.tidx = int(items[0]), int(items[1])
        # Without changing the case
        tok.word, tok.lemma = items[2], items[3]
        tok.pos = items[4]
        tok.deplabel = items[5]
        try:
            tok.hidx = int(items[6])
        except ValueError:
            pass
        tok.ner, tok.partialparse = items[7], items[8]
        if "|" in tok.ner:
            tok.xml, tok.genre, tok.edu_func, tok.edu_depdir = tok.ner.split("|")
        if "|" in tok.partialparse:
            tok.s_type, tok.subjectivity, tok.sentiment, tok.seqlab, tok.seqconf = tok.partialparse.split("|")
            tok.subjectivity = float(tok.subjectivity)
            tok.sentiment = float(tok.sentiment)
            tok.seqconf = float(tok.seqconf)
        try:
            tok.eduidx = int(items[9])
        except ValueError:
            print(str(tok.word) + str(self.fmerge))
            # sys.exit()
            pass
        return tok

    def _recoveredus(self, tokendict):
        """ Recover EDUs from tokendict
        """
        N, edudict = len(tokendict), {}
        for gidx in range(N):
            token = tokendict[gidx]
            eidx = token.eduidx
            try:
                val = edudict[eidx]
                edudict[eidx].append(gidx)
            except KeyError:
                edudict[eidx] = [gidx]
        return edudict


if __name__ == '__main__':
    dr = DocReader()
    fmerge = "../data/training/file1.merge"
    doc = dr.read(fmerge)
    print(len(doc.edudict))
