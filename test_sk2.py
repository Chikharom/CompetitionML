import pandas
import numpy as np
import math
import string
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn import svm

vectorizer = TfidfVectorizer()
# the following will be the training data

trainData=pandas.read_csv("train.csv")

#TestData = pandas.read_csv("test.csv")
#data=[trainData["Abstract"]+TestData["Abstract"], np.array(trainData["Category"])]

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


Clf = svm.SVC()


#text = np.append(np.array(trainData["Abstract"]),np.array(TestData["Abstract"]))
count_vect = CountVectorizer()
Vectors = count_vect.fit_transform(data[0])
X_train = Vectors[:6000]
X_test=Vectors[6000:]
Y_train=data[1][:6000]
Y_test=data[1][6000:]


#Y_train = data[1]

#X_train_counts = count_vect.fit_transform(a)

Clf.fit(X_train,Y_train)

preds =Clf.predict(X_test)

def predictionsToCSV(preds):
	ids=[]
	category=[]
	for i in range(len(preds)):
		ids.append(i)
		category.append(preds[i])
	dik={}
	dik["Category"]=preds
	frame=pandas.DataFrame.from_dict(dik)
	frame.to_csv("predictions", header=True)

temp=0
for i in range(len(Y_test)):
    if Y_test[i]==preds[i]:
        temp+=1

print(temp/(len(preds)))

#predictionsToCSV(preds)