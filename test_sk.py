import pandas
import numpy as np
import math
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import naive_bayes
vectorizer = TfidfVectorizer()
# the following will be the training data
trainData=pandas.read_csv("train.csv")
data=[trainData["Abstract"], np.array(trainData["Category"])]
def split(data,cat):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(len(cat)):
        if i % 5 <4:
            train_data.append(data[i])
            train_labels.append(cat[i])
        else :
            test_data.append(data[i])
            test_labels.append(cat[i])
    return train_data , train_labels,test_data,test_labels
#Clf = naive_bayes.BernoulliNB(alpha=0.4)
Clf = SVC()
count_vect = CountVectorizer()
Vectors = count_vect.fit_transform(data[0])
X_train = Vectors[:6000]
X_test = Vectors[6000:]
Y_train = data[1][:6000]
Y_test = data[1][6000:]
#X_train_counts = count_vect.fit_transform(a)
Clf.fit(X_train,Y_train)
preds =Clf.predict(X_test)
d = Y_test
#X_test_counts = count_vect.fit_transform(c)
sum = 0
for i in range(len(d)):
    if preds[i] == d[i]:
        sum += 1
print(sum/len(d))
