import pandas
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import string
from nltk.corpus import stopwords 

class DataSet:
	
	def __init__(self,seed, mode=0):
		self.seed=seed%5
		self.categories=None
		self.catCount=None
		self.Voc=None
		self.dataMatrix=None
		self.trainSet=None
		self.validSet=None
		self.testSet=None
		if mode==0:
			self.initTest()



	def initTest(self):

		brute=self.getTrainData()
		categories=brute[1]
		abstractSet=brute[0]
		cleaned=self.format(abstractSet)
		dataSet=cleaned

		trainSet, validSet, testSet=self.split_dataset(dataSet)

		Voc=self.genVoc(trainSet, categories)
		priors=self.getCatCount(trainSet, categories)

		matrix=self.genMatrix(trainSet, Voc, categories)
		self.trainSet=trainSet
		self.validSet=validSet
		self.testSet=testSet
		self.matrix=matrix
		self.Voc=Voc
		print(len(self.Voc))
		self.categories=categories
		self.catCount=priors


	def getTrainData(self):
		trainData=pandas.read_csv("train.csv")
		data=[trainData["Abstract"], trainData["Category"]]
		categories={}
		for i in data[1]:
			if i not in categories:
				categories[i]=len(categories)

		return (data,categories)


	def format(self,data):

		lem=WordNetLemmatizer()
		stem=PorterStemmer()

		newFormat=[]
		for i in range(len(data[0])):
			text = data[0][i].split(" ")
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

			text3=[]

			for j in text2:
				#carToClean=["$", "(", ")", "{", "}", "\\", "/", "[", "]", ":", ";", ",", "."]
				carToClean=string.punctuation
				for car in carToClean:
					j=j.replace(car, "")
				text3.append(j)

			text4=[]
			for j in text3:
				if j not in stopwords.words('english'):
					j=stem.stem(j)
					text4.append(j)



			newEntry=[text4, data[1][i]]
			newFormat.append(newEntry)
					
		return newFormat

	def split_dataset(self,dataSet):
	    splitTrain1=[]
	    splitValid1=[]
	    splitTest1=[]
	    for i in range(len(dataSet)):

	        if i%5==self.seed%5:
	            splitValid1.append(dataSet[i])
	        elif i%5==(self.seed+1)%5:
	            splitTest1.append(dataSet[i])
	        else:
	        	splitTrain1.append(dataSet[i])

	    return (splitTrain1, splitValid1, splitTest1)

	def genVoc(self,trainingSet,categories):
		Voc={}
		index=0
		for i in range(len(trainingSet)):
			#print(i)
			for word in trainingSet[i][0]:
				if word not in Voc.keys():
					if len(word)>0:
						Voc[word] = index
						index+=1

		return Voc

	def getCatCount(self,trainSet, categories):
	
		priors=[0]*len(categories)

		for i in range(len(trainSet)):
			priors[categories[trainSet[i][1]]]+=1
		return priors

	def genVector(self,abstract, Voc):
		vector=[0]*len(Voc.keys())
		for word in abstract:
			if word in Voc.keys():
				vector[Voc[word]]+=1
		return vector


	def genMatrix(self,trainSet, Voc, categories):
		
		dataMatrix=[]

		for i in range(len(trainSet)):
			text=trainSet[i][0]
			cat=categories[trainSet[i][1]]
			vector=self.genVector(text, Voc)
			entry=[vector, cat]
			dataMatrix.append(entry)

		return dataMatrix

	def getCategoryFromI(self, index):

		for cat in self.categories.keys():
			if self.categories[cat]==index:
				return cat