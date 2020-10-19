import pandas
import numpy as np
import math
import string

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import naive_bayes


def getTrainData():
	trainData=pandas.read_csv("train.csv")
	data=[trainData["Abstract"], trainData["Category"]]
	categories={}
	for i in data[1]:
		if i not in categories:
			categories[i]=len(categories)

	return (data,categories)

def getTestData():
	trainData=pandas.read_csv("test.csv")
	data=[trainData["Abstract"]]

	return (data)



def cleanup2(data):

	Voc = {}
	for i in range(len(data)):
		text = data[i].split(" ")
		text1=[]
		test="test, test"
		for j in text:
			if "\n" in j or ",":
				temp = j.split("\n")
				for g in temp:
					text1.append(g)
			else:
				text1.append(j)

		text2=[]
		for j in text1:
			if len(j)>=1 and (j[-1]==")" or j[-1]==":" or j[-1]==";" or j[-1]=="."):
				j=j[:-1]
			if len(j)>=1 and (j[0]=="("):
				j=j[1:]
			text2.append(j)

		"""text3=[]
		for j in text2:
			if "$" not in j and "/" not in j and "}" not in j and "{" not in j:
				text3.append(j)"""

		text3=[]

		for j in text2:

			while "$" in j:
				ind=j.index("$")
				if ind==0:
					j=j[1:]
				elif ind==len(j)-1:
					j=j[:-1]
				else:
					j1=j[:ind]
					j2=j[ind+1:]
					j=j1+j2
			text3.append(j)

		data[i]=text3

	return data

def cleanup(data):

	newData=[]
	lem=WordNetLemmatizer()
	stem=PorterStemmer()
	for i in range(len(data)):
		text = data[i].split(" ")
		text1=[]
		test="test, test"
		for j in text:
			if "\n" in j:
				temp = j.split("\n")
				for g in temp:
					text1.append(g)
			else:
				text1.append(j)

		text2=[]
		for j in text1:
			if "-" in j:
				temp = j.split("-")
				for g in temp:
					text2.append(g)
			else:
				text2.append(j)

		"""text3=[]
		for j in text2:
			if "$" not in j and "/" not in j and "}" not in j and "{" not in j:
				text3.append(j)"""

		text3=[]

		for j in text2:
				#carToClean=["$", "(", ")", "{", "}", "\\", "/", "[", "]", ":", ";", ",", "."]
			carToClean=string.punctuation
			numbers = ["0"."1","2"."3","4","5","6","7","8","9"]
			for car in carToClean:
				j=j.replace(car, "")
			for n in numbers:
				j=j.replace(n,"")
			text3.append(j)


		text4=[]
		for j in text3:
			if j not in stopwords.words('english'):
				text4.append(j)

		data[i]=text4

	return data

def genVoc(trainingSet,categories, priors):
	Voc={}
	for i in range(len(trainingSet[0])):
		#print(i)
		readWords = {}
		for word in trainingSet[0][i]:
			if word not in readWords.keys():
				readWords[word]=0
				if word not in Voc.keys():
					Voc[word] = [0]*15
					Voc[word][categories[trainingSet[1][i]]] = 1
				else :
					Voc[word][categories[trainingSet[1][i]]] += 1

	for key in Voc.keys():
		for cat in categories.keys():
			Voc[key][categories[cat]]=(Voc[key][categories[cat]]+1)/(priors[cat]+2)

	return Voc

def getPriors(trainSet, categories):

	priors={}
	for i in categories.keys():
		priors[i]=0

	for i in range(len(trainSet[0])):
		priors[trainSet[1][i]]+=1
	return priors


def split_dataset(dataSet):
    splitTrain1=[]
    splitTrain2=[]
    splitTest1=[]
    splitTest2=[]
    for i in range(len(dataSet[0])):

        if i%5<4:
            splitTrain1.append(dataSet[0][i])
            splitTrain2.append(dataSet[1][i])
        else:
            splitTest1.append(dataSet[0][i])
            splitTest2.append(dataSet[1][i])

    split1=[splitTrain1, splitTrain2]
    split2=[splitTest1, splitTest2]
    return (split1, split2)


def preTreatment(Voc):
	newVoc={}
	for i in Voc.keys():
		if len(i)>0:
			newVoc[i]=Voc[i]
	return newVoc

def preTreatment2(Voc):

	newVoc={}
	var=[]

	for key in Voc.keys():
		probs=np.array(Voc[key])
		var.append(np.var(probs))

	median=np.mean(var)
	std=np.std(var)
	#median=harmonicMean(var)

	for key in Voc.keys():
		probs=np.array(Voc[key])
		if np.var(probs)<median+2.5*std and len(key)>0:
			newVoc[key]=Voc[key]
	return newVoc


def makePredictions(Voc, priors, testSet, categories):
	predictions=[]
	for doc in testSet[0]:
	 	scoreDoc=[0]*len(categories)
	 	readWord={}
	 	for word in doc:
	 		if word not in readWord and word in Voc:
	 			readWord[word]=0
	 			for cat in categories.keys():
	 				scoreDoc[categories[cat]]+=math.log(Voc[word][categories[cat]])

	 	for key in priors.keys():
	 		scoreDoc[categories[key]]+=math.log(priors[key])

	 	pred=None

	 	ind=scoreDoc.index(max(scoreDoc))
	 	for cat in categories.keys():
	 		if ind == categories[cat]:
	 			pred=cat
	 	predictions.append(pred)

	return predictions

def computeGood(testSet, predictions):
	good=0
	for i in range(len(testSet[1])):
		if testSet[1][i]==predictions[i]:
			good+=1
	return good/len(predictions)

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



brute=getTrainData()
categories=brute[1]
abstract=brute[0][0]
cleaned=cleanup(abstract)



"""answers =  brute[0][1]
answers1= []
cleaned2=[]
for i in range(len(answers)):
	answers1.append(answers[i])
	answers1.append(answers[i])
	text1=cleaned[i][:int(len(cleaned[i])/2)]
	text2=cleaned[i][int(len(cleaned[i])/2):]
	cleaned2.append(text1)
	cleaned2.append(text2)
dataSet=[cleaned2, answers1]"""

dataSet=[cleaned, brute[0][1]]


#trainSet=dataSet

trainSet, testSet=split_dataset(dataSet)

priors=getPriors(trainSet, categories)
#Voc=genVoc(trainSet, categories, priors)

for key in priors.keys():
	priors[key]=priors[key]/len(trainSet)

#Voc=preTreatment(Voc)
#print(len(Voc.keys()))

"""temp=getTestData()
abstractTest=temp[0]
cleanedTest=cleanup(abstractTest)
testSet=[cleanedTest]"""
print("here")
trainData=pandas.read_csv("train.csv")
data=[trainData["Abstract"], trainData["Category"]]
Clf = naive_bayes.MultinomialNB()
Clf.fit(data[0],data[1])
preds = Clf.predict(testSet[0])

#preds=makePredictions(Voc, priors, testSet, categories)
print(computeGood(testSet, preds))
#predictionsToCSV(preds)
