"""
======================================================
Create raw text dataset from Reuters
======================================================

This script uses the code from the scikit-learn example 
plot_out_of_core_classification.py for retrieving the Reuters dataset.

The dataset used in this example is Reuters-21578 as provided by the UCI ML
repository. It will be automatically downloaded and uncompressed on first run.
"""

# Authors: Eustache Diemert <eustache@diemert.fr>
#          @FedericoV <https://github.com/FedericoV/>
# License: BSD 3 clause

from __future__ import print_function

from glob import glob
import itertools
import os.path
import re
import tarfile

import numpy as np

from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from sklearn.datasets import get_data_home

import pickle


###############################################################################
# Reuters Dataset related routines
###############################################################################

def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()

class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path,
                                   reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc

def get_minibatch(doc_iter, size, pos_class):
    """Extract a minibatch of examples, return a tuple X_text, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(u'{title}\n\n{body}'.format(**doc), doc['topics'])
            for doc in itertools.islice(doc_iter, size)
            if doc['topics']]

    # If there's no data, just return empty lists.    
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int).tolist()
    
    # Otherwise, retrieve the articles and class labels. zip just splits apart
    # the two variables.
    X_text, y = zip(*data)

    # Convert X_text and y from tuples to lists.    
    X_text = list(X_text)    
    y = list(y)
    
    # Convert the class labels to a list.
    #y = np.asarray(y, dtype=int).tolist()    
    
    # For some reason, some of these articles are just whitespace. Look for 
    # these and remove them. 
    toRemove = []
    docNum = 0
    
    # For each article...
    for article in X_text:
        # If the article is just whitespace, or is empty, we'll remove it        
        if article.isspace() or (article == ""):
            toRemove.append(docNum)
            
        docNum += 1
    
    # Remove the empty articles. Do this in reverse order so as not to corrupt
    # the indeces as we go.
    toRemove.reverse()
    for i in toRemove:
        del X_text[i]
        del y[i]
    
    return X_text, y



def iter_minibatches(doc_iter, minibatch_size):
    """Generator of minibatches."""
    X_text, y = get_minibatch(doc_iter, minibatch_size)
    while len(X_text):
        yield X_text, y
        X_text, y = get_minibatch(doc_iter, minibatch_size)

###############################################################################
# Main
###############################################################################

# Iterator over parsed Reuters SGML files.
data_stream = stream_reuters_documents()

# The Reuter's dataset includes many different classes, but we're just going to
# do binary classification. We'll use 'acq' (articles related to 
# "acquisitions"--one of the most prevalent classes in the dataset) as the 
# positive class, and all other article topics will be used as negative 
# examples.
positive_class = 'acq'

# Retrieve a set of examples from the dataset to use as the training set, then 
# another set of examples to use as the test set. The actual number will
# be smaller because it will exclude "invalid docs with no topics assigned".
X_train_raw, y_train_raw = get_minibatch(data_stream, 5000, positive_class)
X_test_raw, y_test_raw = get_minibatch(data_stream, 5000, positive_class)

print("Train set is %d documents" % (len(y_train_raw)))
print("Test set is %d documents" % (len(y_test_raw)))

# Dump the dataset to a pickle file.
pickle.dump((X_train_raw, y_train_raw, X_test_raw, y_test_raw), open("data/raw_text_dataset.pickle", "wb"))
