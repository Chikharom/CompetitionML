from dataSet import DataSet
from perceptron import PerceptronBasic
data=DataSet(19)
perc=PerceptronBasic(data)
perc.train()
preds=perc.predictOnTest()
print(perc.computeGood(preds))

