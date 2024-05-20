import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# 1. K-Nearest Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(features_train,labels_train)
accuracy = clf.score(features_test,labels_test)
print("K-Nearest Neighbors Algorithm Accuracy:",accuracy)



# 2. Adaptive Boosting Algorithm 
from sklearn.ensemble import AdaBoostClassifier

clf1 = AdaBoostClassifier(algorithm='SAMME')
clf1 = clf1.fit(features_train,labels_train)
accuracy1 = clf1.score(features_test,labels_test)
print("Adaptive Boosting Algorithm Accuracy:",accuracy1)



# 3. Random Forests Algorithm
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier()
clf2 = clf2.fit(features_train,labels_train)
accuracy2 = clf2.score(features_test,labels_test)
print("Random Forests Algorithm Accuracy:",accuracy2)



# 4. Decision Tree Algorithm
from sklearn import tree

clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(features_train,labels_train)
accuracy3 = clf3.score(features_test,labels_test)
print("Decision Tree Algorithm Accuracy:",accuracy3)

# 5. SVM Algorithm
from sklearn.svm import SVC

clf4 = SVC(kernel='rbf')
clf4 = clf4.fit(features_train,labels_train)
accuracy4 = clf4.score(features_test,labels_test)
print("SVM Algorithm Accuracy:",accuracy4)

# 6. NB Algorithm
from sklearn.naive_bayes import GaussianNB

clf5 = GaussianNB()
clf5 = clf5.fit(features_train,labels_train)
accuracy5 = clf5.score(features_test,labels_test)
print("NB Algorithm Accuracy:",accuracy5)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
