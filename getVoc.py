import pandas
import numpy as np
import math


def getTrainData():
	trainData=pandas.read_csv("train.csv")
	data=[trainData["Abstract"], trainData["Category"]]
	categories={}
	for i in data[1]:
		if i not in categories:
			categories[i]=len(categories)

	return (data,categories)


def cleanup(data):

	Voc = {}
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

			carToClean=["$", "(", ")", "{", "}", "\\", "/", "[", "]", ":", ";", ",", ".", "\"", "*","\'"]

			for car in carToClean:

				j=j.replace(car, "")

			j=j.lower()

			text3.append(j)

		data[i]=text3
				
	return data

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
		if len(i)>5:
			newVoc[i]=Voc[i]
	return newVoc


def preTreatment2(Voc):
	
	newVoc={}
	var=[]

	for key in Voc.keys():
		probs=np.array(Voc[key])
		var.append(np.var(probs))

	median=np.median(var)
	#median=harmonicMean(var)

	for key in Voc.keys():
		probs=np.array(Voc[key])
		if np.var(probs)<median:
			newVoc[key]=Voc[key]
	return newVoc


def genVoc(trainingSet,categories):
	Voc={}
	index=0
	for i in range(len(trainingSet[0])):
		#print(i)
		for word in trainingSet[0][i]:
			if word not in Voc.keys():
				if len(word)>4:
					Voc[word] = index
					index+=1

	return Voc

def getCatCount(trainSet, categories):
	
	priors=[0]*len(categories)

	for i in range(len(trainSet[0])):
		priors[categories[trainSet[1][i]]]+=1
	return priors

def genVector(abstract, Voc):
	vector=[0]*len(Voc.keys())
	for word in abstract:
		if word in Voc.keys():
			vector[Voc[word]]+=1
	return vector


def genMatrix(trainSet, Voc, categories, priors):
	
	dataMatrix=[]
	for i in range(len(categories)):
		dataMatrix.append([])

	for i in range(len(trainSet[0])):
		text=trainSet[0][i]
		cat=trainSet[1][i]
		vector=genVector(text, Voc)
		dataMatrix[categories[cat]].append(vector)

	newData=[]

	for i in range(len(dataMatrix)):
		newVec=np.array([0]*len(Voc))
		for j in range(len(dataMatrix[i])):
			newVec+=np.array(dataMatrix[i][j])

		newVec=newVec/priors[i]
		newVec=newVec/(np.linalg.norm(newVec))
		newData.append(newVec)
	return newData

def getData1():

	brute=getTrainData()
	categories=brute[1]
	abstract=brute[0][0]
	cleaned=cleanup(abstract)
	dataSet=[cleaned, brute[0][1]]

	trainSet, testSet=split_dataset(dataSet)

	Voc=genVoc(trainSet, categories)
	priors=getCatCount(trainSet, categories)

	matrix=genMatrix(trainSet, Voc, categories, priors)
	return trainSet, testSet, matrix, Voc, categories


def makePredictions(matrix, testSet, categories, Voc):
	predictions=[]
	for doc in testSet[0]:
	 	
	 	dotProd=[0]*len(categories)
	 	vector=genVector(doc, Voc)
	 	
	 	for cat in categories.keys():
	 		dotProd[categories[cat]]=np.dot(vector, matrix[categories[cat]])

	 	ind=dotProd.index(max(dotProd))
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

trainSet, testSet, matrix, Voc, categories=getData1()
print(Voc.keys())

preds=makePredictions(matrix, testSet, categories, Voc)
print(computeGood(testSet, preds))
