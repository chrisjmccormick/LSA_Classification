# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:04:40 2015

@author: Chris
"""

import pickle
import time
import numpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

# The raw text dataset is stored as tuple in the form:
# (X_train_raw, y_train_raw, X_test_raw, y_test)
# The 'filtered' dataset excludes any articles that we failed to retrieve
# fingerprints for.
raw_text_dataset = pickle.load( open( "data/raw_text_dataset.pickle", "rb" ) )
X_train_raw = raw_text_dataset[0]
y_train = raw_text_dataset[1] 

print("  %d training examples (%d positive)" % (len(y_train), sum(y_train)))

###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################

# Tfidf vectorizer:
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
#     document length on the tf-idf values. 
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it 
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)

print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

# Get the words that correspond to each of the features.
feat_names = vectorizer.get_feature_names()

# Print ten arbitrary feature names
print("Some words in the vocabulary:")
for i in range(1000, 1010):
    print "  ", feat_names[i]
    
print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first N principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)

# The SVD matrix will have one row per component, and one column per feature
# of the original data.
# Get the first component.

for compNum in range(0, 100, 10):

    comp = svd.components_[compNum]
    
    # Sort the weights in the first component, and get the indeces
    indeces = numpy.argsort(comp).tolist()
    
    # Reverse the indeces, so we have the largest weights first.
    indeces.reverse()
    
    # Print the strongest 10 features in the component.
    print("\nTop 10 features for component %d:" % (compNum))
    for i in range(0, 10):
        weightIndex = indeces[i]    
        print("  %s    %.3f" % (feat_names[weightIndex], comp[weightIndex]))

#print("  done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("\nExplained variance of the SVD step: {}%".format(int(explained_variance * 100)))


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nClassifying tfidf vectors...")

