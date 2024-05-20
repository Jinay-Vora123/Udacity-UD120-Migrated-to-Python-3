t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
clf.fit(features_test, labels_test)
print("Predicting Time:", round(time()-t0, 3), "s")