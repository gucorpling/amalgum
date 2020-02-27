## readdoc.py
## Author: Yangfeng Ji
## Date: 02-14-2015
## Time-stamp: <yangfeng 02/14/2015 18:29:54>

from os import listdir
from os.path import join
from .docreader import DocReader
import gzip
import sys

PY3 = sys.version_info[0] > 2

if PY3:
    from pickle import dump
else:
    from cPickle import dump

def readdoc(rpath, fdocdict):
    files = [join(rpath, fname) for fname in listdir(rpath) if fname.endswith(".merge")]
    dr = DocReader()
    docdict = {}
    for fmerge in files:
        doc = dr.read(fmerge)
        docdict[fmerge] = doc
    print('Write doc dict into {}'.format(fdocdict))
    with gzip.open(fdocdict, 'w') as fout:
        dump(docdict, fout)
