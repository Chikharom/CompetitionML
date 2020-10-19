import pandas
import numpy as np
import math

class multinomialBayes:

	def __init__(self, dataSet):

		self.dataSet=dataSet
		self.conditionalProbs=self.formProbs()
		self.priors=self.computePriors()

	def formProbs(self):

		matrix=self.dataSet.matrix
		conditionalProbs=np.zeros([len(self.dataSet.categories), len(self.dataSet.Voc)])

		for pair in matrix:
			conditionalProbs[pair[1]]=np.add(conditionalProbs[pair[1]], pair[0])

		for line in conditionalProbs:
			s=np.sum(line)
			for i in range(len(line)):
				line[i]=(line[i]+1)/(s+2)

		return conditionalProbs

	def computePriors(self):

		total=len(self.dataSet.trainSet)
		priors=[]
		for i in self.dataSet.catCount:
			priors.append(i/total)

		return priors

	def predict(self, abstract):

		voc=self.dataSet.Voc
		cat=self.dataSet.categories
		
		score=[0]*len(cat)
		
		for word in abstract:

			if word in voc.keys():
			
				j=voc[word]
			
				for c in cat.keys():
					score[cat[c]]+=math.log(self.conditionalProbs[cat[c]][j])

		for c in cat.keys():
			score[cat[c]]+=self.priors[cat[c]]

		return score.index(max(score))

	def predictOnTest(self):
		predictions=[]
		temp=0
		for pair in self.dataSet.testSet:
			pred=self.predict(pair[0])
			textPred=self.dataSet.getCategoryFromI(pred)
			predictions.append(textPred)
			if self.dataSet.compare(pair[1], textPred):
				temp+=1
		return temp/len(predictions)


