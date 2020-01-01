
import csv
from collections import defaultdict
import pydotplus
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.utils import shuffle


class DecisionTree:
    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None, summary=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results 
        self.summary = summary


def kfold(data,k):

    X= shuffle(data,random_state=42)
    X=X.to_numpy()
    n = len(data)/k
    if(n>int(n)):
        n= (int(n)+1)
    trainingData = X[0:n*(k-1)]
    test = X[n*(k-1):len(data)]
    
    testData = test[:,:test.shape[1]-1]
    y_test = test[:,test.shape[1]-1:]
    decisionTree = growTree(trainingData, evaluationFunction=gini)
    prune(decisionTree, 0.8, notify=True) 
    count=0
    count1=0
    true=[]
    pred=[]
    for i in range(testData.shape[0]):
        count1 +=1
        t = classify(testData[i], decisionTree)
        for key, value in t.items():
            
            pred.append(key)
            true.append(y_test[i])
            if(key==y_test[i]):
                count +=1
    
    print("\nPredictive accuracy for k = ",k," is ",count/count1)
    print(confusion_matrix(true,pred))
    a,b,c,d = precision_recall_fscore_support(true, pred, average="macro")
    print("Precision = ",a, "\nRecall = ",b," \nF1-score = ",c)


def bootstrap(data,n):
    data = data.to_numpy()
    for j in range(n):
        trainingData = resample(data,n_samples=250)
        testData = resample(data,n_samples=50)
        y_test = testData[:,testData.shape[1]-1:]
        testData = testData[:,:testData.shape[1]-1]
        decisionTree = growTree(trainingData, evaluationFunction=gini)
        prune(decisionTree, 0.8, notify=True) 
        count=0
        count1=0
        true=[]
        pred=[]
        for i in range(testData.shape[0]):
            count1 +=1
            t = classify(testData[i], decisionTree)
            for key, value in t.items():

                pred.append(key)
                true.append(y_test[i])
                if(key==y_test[i]):
                    count +=1
    
        print("\nPredictive accuracy for Bootstrap = ",j+1," is ",count/count1)
        print(confusion_matrix(true,pred))
        a,b,c,d = precision_recall_fscore_support(true, pred, average='macro')
        print("Precision = ",a, "\nRecall = ",b,"\nF1-score = ",c)
        

def Unique_Counts(rows):
    results_ = {}
    for row in rows:
        r = row[-1]
        if r not in results_: results_[r] = 0
        results_[r] += 1
    return results_


def entropy(rows):
    from math import log
    log2 = lambda x: log(x)/log(2)
    results_ = Unique_Counts(rows)
    entropy_value = 0.0
    for r in results_:
        prob = float(results_[r])/len(rows)
        entropy_value -= prob*log2(prob)
    return entropy_value


def divideSet(trows, column_, val):
    splitFn = None
    if isinstance(val, int) or isinstance(val, float): 
        splitFn = lambda row : row[column_] >= val
    else: 
        splitFn = lambda row : row[column_] == val

    lista = [row for row in trows if splitFn(row)]
    listb = [row for row in trows if not splitFn(row)]
    return (lista, listb)


def gini(trows):
    total = len(trows)
    count = Unique_Counts(trows)
    imp_val = 0.0

    for ka in count:
        pa = float(count[ka])/total

        for kb in count:
            if ka == kb: continue
            pb = float(count[kb])/total
            imp_val += (pa*pb)

    return imp_val


def growTree(rows, evaluationFunction=entropy):
    if len(rows) == 0: return DecisionTree()
    currScore = evaluationFunction(rows)

    gain_best = 0.0
    bestAttribute = None
    bestSets = None

    columnCount = len(rows[0]) - 1  
    for col_ in range(0, columnCount):
        columnValues = [row_[col_] for row_ in rows]
        lsUnique = list(set(columnValues))

        for value in lsUnique:
            (seta, setb) = divideSet(rows, col_, value)

            prob = float(len(seta)) / len(rows)
            gain = currScore - prob*evaluationFunction(seta) - (1-prob)*evaluationFunction(setb)
            if gain>gain_best and len(seta)>0 and len(setb)>0:
                gain_best = gain
                bestAttribute = (col_, value)
                bestSets = (seta, setb)

    dcY = {'impurity' : '%.3f' % currScore, 'samples' : '%d' % len(rows)}
    if gain_best > 0:
        trueBranch = growTree(bestSets[0], evaluationFunction)
        falseBranch = growTree(bestSets[1], evaluationFunction)
        return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch,
                            falseBranch=falseBranch, summary=dcY)
    else:
        return DecisionTree(results=Unique_Counts(rows), summary=dcY)


def prune(tree, minGain, evaluationFunction=entropy, notify=False):

    if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
    if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        ta, fa = [], []

        for v_, c_ in tree.trueBranch.results.items(): ta += [[v_]] * c_
        for v_, c_ in tree.falseBranch.results.items(): fa += [[v_]] * c_

        prob = float(len(ta)) / len(ta + fa)
        delta_val = evaluationFunction(ta+fa) - prob*evaluationFunction(ta) - (1-prob)*evaluationFunction(fa)
        if delta_val < minGain:
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = Unique_Counts(ta + fa)


def classify(obs, tree):

    def classify_(obs, tree):
        if tree.results != None:  
            return tree.results
        else:
            val = obs[tree.col]
            branch_ = None
            if isinstance(val, int) or isinstance(val, float):
                if val >= tree.value: branch_ = tree.trueBranch
                else: branch_ = tree.falseBranch
            else:
                if val == tree.value: branch_ = tree.trueBranch
                else: branch_ = tree.falseBranch
        return classify_(obs, branch_)

    return classify_(obs, tree)

        
if __name__ == '__main__':
        from sklearn.metrics import confusion_matrix

        #dataset in the same directory as the code file        
        data=pd.read_csv("SOYABEAN.csv",header=None,index_col=None)
        target = data.iloc[:,0]
        data = data.drop(data.columns[0],axis = 1)
        data = data.assign(target1=target)
        data.columns = range(data.shape[1])
        
        for i in range(2,13):
            kfold(data,i)
        bootstrap(data,1)

