from dataSet import DataSet
from perceptron import PerceptronBasic
from sklearn import svm
from multiBayes import multinomialBayes
from reducedDataSet import reducedDataSet

def mainPerc():
	data=DataSet(16)
	data.reduceWords(250)
	perc=PerceptronBasic(data)
	perc.train(10*len(data.trainSet))
	preds=perc.predictOnTest()
	print(perc.computeGood(preds))

def simpleSVM():

	data=DataSet(19)

	X=[]
	y=[]
	matrixTrain=data.matrix

	for pair in matrixTrain:
		X.append(pair[0])
		y.append(pair[1])

	matrixTest=data.genTestMatrix(data.testSet, data.Voc, data.categories)
	Xtest=[]

	for pair in matrixTest:
		Xtest.append(pair[0])


	clf=svm.SVC()
	clf.fit(X, y)
	pred=clf.predict(Xtest)

	print(data.computeGood(pred))


def bayes():
	data=DataSet(16)
	data.reduceWords(4000)
	bayes=multinomialBayes(data)
	print(bayes.predictOnTest())

def testCount():
	data=DataSet(16)
	n=500
	maxWords=data.getNmax(n, dictMax=True)
	overlap=[]
	read=[]
	for cat in data.categories:
		read.append(cat)
		for cat2 in data.categories:
			if cat2 in read:
				continue
			temp=0
			for word in maxWords[cat]:
				if word in maxWords[cat2]:
					temp+=1
			overlap.append([temp/n, cat+" "+cat2])

	sorted=False
	while not sorted:
		sorted=True
		for i in range(len(overlap)-1):
			if overlap[i][0]>overlap[i+1][0]:
				sorted=False
				temp=overlap[i]
				overlap[i]=overlap[i+1]
				overlap[i+1]=temp
	print(overlap)

def reducedClassifier():
	catDict={"astro-ph": "ASTRO", "astro-ph.CO":"ASTRO", "astro-ph.SR":"ASTRO", "astro-ph.GA":"ASTRO",
	"cs.LG":"CS", "stat.ML":"CS", 
	"math.CO":"MATH", "math.AP":"MATH",
	"physics.optics":"COND", "quant-ph":"COND", "cond-mat.mtrl-sci":"COND", "cond-mat.mes-hall":"COND",
	"gr-qc":"RELAT", "hep-th":"RELAT", "hep-ph":"RELAT"}
	data=reducedDataSet(16, 2000, catDict)
	bayes=multinomialBayes(data)
	print(bayes.predictOnTest())


reducedClassifier()

#testCount()

#bayes()
#mainPerc()