import sys
from time import time
sys.path.append(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools")
import email_preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess()


#########################################################
### your code goes here ###

features_train1 = features_train[:int(len(features_train)/100)]
labels_train1 = labels_train[:int(len(labels_train)/100)]

features_test1 = features_test[:int(len(features_test)/100)]
labels_test1 = labels_test[:int(len(labels_test)/100)]

clf = SVC(kernel='rbf', C=10)

t0 = time()
clf.fit(features_train1,labels_train1)
print("Training Time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test1)
print("Predicting Time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_test1, pred)
print("Accuracy:",accuracy)
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''



#########################################################
