import sys
import os
from time import time
sys.path.append(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools")
import email_preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
clf.fit(features_test, labels_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = clf.score(features_test,labels_test)
print("Accuracy:", accuracy)
##############################################################