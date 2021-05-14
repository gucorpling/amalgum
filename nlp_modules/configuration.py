# RST model location
MODEL_SERVER = "https://corpling.uis.georgetown.edu/amir/download/"
RSTDT_MODEL_PATH = "amalgum/rst/rstdt_collapsed.pt"
XML_ATTRIB_REFDATE = "dateCreated"  # reference date for date/time recognizer
XML_ATTRIB_COLLECTDATE = "dateCollected"
XML_ROOT_TIMEX3 = "TimeML"
HEIDELTIME_STANDALONE = "https://github.com/HeidelTime/heideltime/releases/download/VERSION2.2.1/heideltime-standalone-2.2.1.tar.gz"
TREETAGGER_LINUX = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.3.tar.gz"
TREETAGGER_WINDOWS = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-windows-3.2.3.zip"
TREETAGGER_MACOSX = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-3.2.3.tar.gz"

# scripts and configuration models are same for all 3 OS
TREETAGGER_SCRIPTS = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz"
TREETAGGER_EXEC = (
    "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh"
)
TREETAGGER_PARAMETER_FILES_PENN = (
    "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz"
)
TREETAGGER_PARAMETER_FILES_BNC = (
    "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-bnc.par.gz"
)
TREETAGGER_CHUNKER = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-chunker.par.gz"

# Location of clone of GUM and EWT repos (only needed for retraining tools)
GUM_ROOT = ""
EWT_ROOT = ""

# leave it to the user to install the Perl interpreter for windows? (Date/Time TreeTagger binary dependency)
# Tested on Windows 10, didn't need to install Perl
# ACTIVEPERL_MSI = 'https://www.activestate.com/products/perl/downloads/thank-you/?dl=https://cli-msi.s3.amazonaws.com/ActivePerl-5.28.msi'
