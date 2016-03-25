"""
======================================================
Classification experiment
======================================================

TODO.....

The dataset used in this example is Reuters-21578 as provided by the UCI ML
repository. It will be automatically downloaded and uncompressed on first run.

"""

# Authors: Eustache Diemert <eustache@diemert.fr>
#          @FedericoV <https://github.com/FedericoV/>
# License: BSD 3 clause

from __future__ import print_function

import time
import pickle

from cortical.client import ApiClient
from cortical.textApi import TextApi


###############################################################################
#  Retina API
###############################################################################
print("Initializing retina API")

RETINA_NAME = "en_associative"

# Create a Retina API client.
client = ApiClient(apiKey="1eb3fc10-5288-11e5-9e69-03c0722e0d16", apiServer="http://api.cortical.io/rest")
textApi = TextApi(client)

def getFingerprintsForArticles(articles):
    """Retrieve fingerprints for the given articles using Cortical.io Retina"""
    
    # List to store the calculated fingerprints.
    fingerprints = []
    
    # List to store indeces of articles for which we failed to retrieve a
    # fingerprint.
    failures = []    
    
    docNum = 0    
    numArticles = len(articles)
    
    # For each of the input text articles...
    for inputText in articles:
        # Print our progress.        
        print("%5d / %5d" % (docNum + 1, numArticles))

        # Get the fingerprint for this article.        
        try:
            fpList = textApi.getRepresentationForText(RETINA_NAME, inputText)
            fingerprints.append(fpList[0].positions)
        except:
            # If it fails, record the index of the document.
            failures.append(docNum)
            print("Failed to retrieve fingerprint for input text:")
            print("|", inputText, "|")
            
        docNum += 1

    return fingerprints, failures


###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

# The raw text dataset is stored as tuple in the form:
# (X_train_raw, y_train_raw, X_test_raw, y_test_raw)
raw_text_dataset = pickle.load( open( "data/raw_text_dataset.pickle", "rb" ) )
X_train_raw = raw_text_dataset[0]
y_train_raw = raw_text_dataset[1] 
X_test_raw = raw_text_dataset[2]
y_test_raw = raw_text_dataset[3]

###############################################################################
#  Retrieve Cortical.io fingerprints for the articles
###############################################################################

tick = time.time()

print("Requesting fingerprints for training articles...")

# Get the fingerprints for the training documents. This will actually make a
# call to Cortical.io's servers to do the processing.
X_train_fp, failures = getFingerprintsForArticles(X_train_raw)

# Deal with failed documents--remove them from the dataset.
# Delete the items in reverse order so that we don't corrupt the indeces.
failures.reverse()
for i in failures:
    del X_train_raw[i]    
    del y_train_raw[i]

y_train_fp = y_train_raw

print("After removing %d failed articles, training set is %d documents (%d positive)" % (len(failures), len(y_train_raw), sum(y_train_raw)))

print("Requesting fingerprints for test articles...")

# Get the fingerprints for the training documents. This will actually make a
# call to Cortical.io's servers to do the processing.
X_test_fp, failures = getFingerprintsForArticles(X_test_raw)

# Deal with failed documents--remove them from the dataset.
# Delete the items in reverse order so that we don't corrupt the indeces.
failures.reverse()
for i in failures:
    del X_test_raw[i]    
    del y_test_raw[i]

y_test_fp = y_test_raw

print("After removing %d failed articles, test set is %d documents (%d positive)" % (len(failures), len(y_test_raw), sum(y_test_raw)))

vectorizing_time = time.time() - tick

print("Time to get all fingerprints: ", vectorizing_time)

# Dump the filtered (without failed articles) raw text dataset to a pickle file.
pickle.dump((X_train_raw, y_train_raw, X_test_raw, y_test_raw), open("data/raw_text_dataset_filtered.pickle", "wb"))

# Dump the calculated fingerprints to a pickle file.
pickle.dump((X_train_fp, y_train_fp, X_test_fp, y_test_fp), open("data/fingerprint_dataset.pickle", "wb"))
