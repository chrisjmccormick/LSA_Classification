# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:04:40 2015

@author: Chris
"""

import pickle
import time

#==============================================================================
#  Load the dataset.

print("Loading dataset...")

# Load the dataset (with pre-computed fingerprints).
X_train = pickle.load( open( "X_train.p", "rb" ) )
y_train = pickle.load( open( "y_train.p", "rb" ) ) 

X_test = pickle.load( open( "X_test.p", "rb" ) )
y_test = pickle.load( open( "y_test.p", "rb" ) )

print("%d training examples (%d positive)" % (len(y_train), sum(y_train)))
print("%d test examples (%d positive)" % (len(y_test), sum(y_test)))

#==============================================================================
#  

# Time this step.
t0 = time.time()

numRight = 0;

curTestIndex = 0;

# Classify each of the test vectors using the nearest neighbor.
for x1 in X_test:
    x1 = set(x1)

    print("%d / %d" % (curTestIndex, len(y_test)))

    # Create an array to hold the Hamming distance.
    HD = []

    # Compare it to each of the training vectors.
    for x2 in X_train:
        x2 = set(x2)
        
        # Calculate the Hamming distance and add it to the list.    
        HD.append(len(x1 ^ x2))
    
    # Get the index of the document with the lowest distance.
    closestIndex = HD.index(min(HD))
    
    # Check if the test example was correctly classified.
    if (y_test[curTestIndex] == y_train[closestIndex]):
        numRight = numRight + 1
    
    curTestIndex = curTestIndex + 1

    #print("Closest matching document %d, JS %.2f, cat: %d" %(closestIndex, JS[closestIndex], y_train[closestIndex]) )

print("(%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
        
print "\nClassification took %.2fsec" % elapsed