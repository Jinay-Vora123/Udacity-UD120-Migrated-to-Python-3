"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\final_project\final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

sort_keys = r'C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools\python2_lesson13_keys.pkl'

### it's all yours from here forward! 
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier()
clf1.fit(features,labels)
print("Accuracy of Overfitted Decision Tree:", clf1.score(features,labels))


from sklearn.model_selection import train_test_split 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
clf2 = DecisionTreeClassifier()
clf2.fit(features_train,labels_train)
print("Accuracy of Test Data having 30 percent of the full data:", clf2.score(features_test,labels_test))