import sys
import joblib
import numpy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

# sys.path.append(".")
# def wdpkl(x):
#     with open(x,'rb') as f:
#         return f

# def eapkl(x):
#     with open(x,'rb') as f:
#         return f

def preprocess(words_file = r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools\word_data.pkl", authors_file = r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\tools\email_authors.pkl"):

    ### the words (features) and authors (labels), already largely preprocessed
    ### this preprocessing will be repeated in the text learning mini-project
    authors_file_handler = open(authors_file, "rb")
    authors = joblib.load(authors_file_handler)


    words_file_handler = open(words_file, "rb")
    word_data = joblib.load(words_file_handler)

    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)


    ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)



    ### feature selection, because text is super high dimensional and 
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    ### info on the data
    print("No. of Chris training emails : ", sum(labels_train))
    print("No. of Sara training emails : ", len(labels_train)-sum(labels_train))
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
