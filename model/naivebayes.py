import pandas as pd
import json

def count(data,colname,label,target):
        cnt=0
        for doc in range(len(data)):
            for val in range(len(data[doc][colname])):
                #print('data[',doc,'][',colname,'][',val,'==',label,' and data[gold_labels][',val,']==',target)
                if data[doc][colname][val]==label and data[doc]['gold_labels'][val]==target:
                    cnt+=1
        return cnt

class NaiveBayes:
    def __init__(self,train_X,test_X,feature_attr_name,labels):    
        self.labels=labels
        self.probabilities = {0:{},1:{}}
        self.train_X=train_X
        self.test_X=test_X
        self.feature_attr_name=feature_attr_name

    def fit(self):
        self.count_0 = count(self.train_X,'gold_labels',0,0)
        self.count_1 = count(self.train_X,'gold_labels',1,1)    
        for col in self.feature_attr_name:
                self.probabilities[0][col] = {}
                self.probabilities[1][col] = {}
                self.prob_0 = self.count_0/(self.count_0+self.count_1)
                self.prob_1 = self.count_1/(self.count_0+self.count_1)
                for category in self.labels:
                    #print('category=',category,'self.labels=',self.labels)
                    self.count_ct_0 = count(self.train_X,col,category,0)
                    self.count_ct_1 = count(self.train_X,col,category,1)
            
                    self.probabilities[0][col][category] = self.count_ct_0 / self.count_0
                    self.probabilities[1][col][category] = self.count_ct_1 / self.count_1
                    
    def predict(self):
        #self.prediction=[]
        self.predicted=[]
        for doc in range(len(self.test_X)):
            #self.predicted=[]
            for sentence in range(len(self.test_X[doc]['gold_labels'])):
                    self.prod_0 = self.prob_0
                    self.prod_1 = self.prob_1
                    for feature in self.feature_attr_name:
                        self.prod_0 *= self.probabilities[0][feature][self.test_X[doc][feature][sentence]]
                        self.prod_1 *= self.probabilities[1][feature][self.test_X[doc][feature][sentence]]
        
                    if self.prod_0 > self.prod_1:
                        self.predicted.append(0)
                    else:
                        self.predicted.append(1)
            #self.prediction.append(self.predicted)
        return self.predicted
        #return self.prediction
        
    def evaluate(self):
        self.tp,self.tn,self.fp,self.fn = 0,0,0,0
        self.plabel=[]
        for doc in range(len(self.test_X)):
            for sentence in range(len(self.test_X[doc]['gold_labels'])):
                self.plabel.append(self.test_X[doc]['gold_labels'][sentence])
        for sentence in range(len(self.plabel)):
            if self.predicted[sentence] == 0:
                if self.test_X[sentence] == 0:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if self.test_X[sentence] == 1:
                    self.tn += 1
                else:
                    self.fn += 1
        #self.num_of_sentence=[val for sublist in self.prediction for val in sublist]
        return print('Accuracy = ',((self.tp+self.tn)/len(self.plabel))*100)
