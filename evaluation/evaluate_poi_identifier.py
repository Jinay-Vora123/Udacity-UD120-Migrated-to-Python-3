"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import numpy
from sklearn.metrics import precision_score, recall_score
sys.path.append(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\final_project\final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

sort_keys = r'C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools\python2_lesson14_keys.pkl'


### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
print("Accuracy of Test Data having 30 percent of the full data:", clf.score(features_test,labels_test))

pred = clf.predict(features_test)
arr,counts = numpy.unique(pred,return_counts=True)
print("Number of POIs in Test Data:",len(counts))
print("Number of people in total:",len(labels_test))
print("Accuracy if POIs=0:",counts[0]/len(features_test))

truePositives=0
for actual, predicted in zip(features_test,labels_test):
    if actual==1 and predicted==1:
        truePositives+=1

print("True Posititives:",truePositives)
print("Precision Score:",precision_score(labels_test,pred))
print("Recall Score:",recall_score(labels_test,pred))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

TruePositives=0
TrueNegatives=0
FalsePositives=0
FalseNegatives=0

for a,p in zip(predictions,true_labels):
    if a==1 and p==1:
        TruePositives+=1
    
    elif a==0 and p==0:
        TrueNegatives+=1
    
    elif a==1 and p==0:
        FalsePositives+=1

    else:
        FalseNegatives+=1

print("True Positives for the second case:",TruePositives)
print("True Negatives for the second case:",TrueNegatives)
print("False Positives for the second case:",FalsePositives)
print("False Negatives for the second case:",FalseNegatives)
print("Precision for the second case:", TruePositives/(TruePositives+FalsePositives))
print("Recall for the second case:", TruePositives/(TruePositives+FalseNegatives))