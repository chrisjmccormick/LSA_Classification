# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:04:40 2015

@author: Chris
"""

import pickle
import time

from sklearn.neighbors import KNeighborsClassifier

###############################################################################
#  Load the fingerprinted dataset.
###############################################################################

print("Loading dataset...")

# The fingerprinted dataset is stored as tuple in the form:
# (X_train_fp, y_train, X_test_fp, y_test)
fingerprint_dataset = pickle.load( open( "data/fingerprint_dataset.pickle", "rb" ) )
X_train_fp = fingerprint_dataset[0]
y_train = fingerprint_dataset[1] 
X_test_fp = fingerprint_dataset[2]
y_test = fingerprint_dataset[3]

print("  %d training examples (%d positive)" % (len(y_train), sum(y_train)))
print("  %d test examples (%d positive)" % (len(y_test), sum(y_test)))


###############################################################################
#  Convert fingerprints from a list of positions to a bit vector
##############################################################################

# The fingerprints are initially a list of bit indeces [3, 6, 7, 8, ...]
# To compare them with distance functions, we want a bit-vector of the form
# [0, 0, 0, 1, 0, 0, 1, 1, 1]
def setToBitArray(x):
    ba = [0]*16384
    for val in x:
        ba[val] = 1
        
    return ba

    
X_train_bv = []    
for x in X_train_fp:
    X_train_bv.append(setToBitArray(x))
    
X_test_bv = []    
for x in X_test_fp:
    X_test_bv.append(setToBitArray(x))


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nClassifying fingerprint vectors with cosine distance...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the hamming distance, 
# and brute-force calculation of distances.
knn_tfidf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_tfidf.fit(X_train_bv, y_train)

# Classify the test vectors.
p = knn_tfidf.predict(X_test_bv)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)


print("\nClassifying fingerprint vectors with Hamming distance...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the hamming distance, 
# and brute-force calculation of distances.
knn_tfidf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='hamming')
knn_tfidf.fit(X_train_bv, y_train)

# Classify the test vectors.
p = knn_tfidf.predict(X_test_bv)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)





