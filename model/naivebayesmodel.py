import pandas as pd
labels=['low','medium','high']
import json
data=[]
with open("D:\ML Project\CleanGithub\Predictator-master\\analysis\\feature_set.jsonl") as f:
    for lines in f:
        data.append(json.loads(lines))
for doc in range(len(data)):    
    for val1 in range(len(data[doc]['gold_labels'])):
        for val2 in range(len(data[doc]['gold_labels'][val1])):
            data[doc]['gold_labels'][val1][val2]=int(data[doc]['gold_labels'][val1][val2])
for doc in range(len(data)):    
    for feature in ('F5','F9','F10','F11','F12','gold_labels'):
        data[doc][feature]=[val for sublist in data[doc][feature] for val in sublist]
for doc in range(len(data)):    
    for feature in ('F1','F2','F3','F5','F6','F7','F9','F10','F11','F12'):
        mean=sum(data[doc][feature])/len(data[doc][feature])
        for element in range(len(data[doc][feature])):
            if data[doc][feature][element]==0:
               data[doc][feature][element]=mean
        data[doc][feature]=pd.cut(data[doc][feature],bins=len(labels),labels=labels)
def count(data,colname,label,target):
    cnt=0
    for doc in range(len(data)):
        for val in range(len(data[doc][colname])):
            if data[doc][colname][val]==label and data[doc]['gold_labels'][val]==target:
                cnt+=1
    return cnt
probabilities = {0:{},1:{}}
train_percent = 70
train_len = int((train_percent*len(data))/100)
train_X = data[:train_len]
test_X = data[train_len+1:]
count_0 = count(train_X,'gold_labels',0,0)
count_1 = count(train_X,'gold_labels',1,1)    
prob_0 = count_0/(count_0+count_1)
prob_1 = count_1/(count_0+count_1)

for col in ('F1','F2','F3','F5','F6','F7','F9','F10','F11','F12'):
        probabilities[0][col] = {}
        probabilities[1][col] = {}
        
        for category in labels:
            count_ct_0 = count(train_X,col,category,0)
            count_ct_1 = count(train_X,col,category,1)
            
            probabilities[0][col][category] = count_ct_0 / count_0
            probabilities[1][col][category] = count_ct_1 / count_1
prediction=[]
for doc in range(len(test_X)):
    predicted=[]
    for sentence in range(len(test_X[doc]['gold_labels'])):
            prod_0 = prob_0
            prod_1 = prob_1
            for feature in ('F1','F2','F3','F5','F6','F7','F9','F10','F11','F12'):
                prod_0 *= probabilities[0][feature][test_X[doc][feature][sentence]]
                prod_1 *= probabilities[1][feature][test_X[doc][feature][sentence]]
        
            #Predict the outcome
            if prod_0 > prod_1:
                predicted.append(0)
            else:
                predicted.append(1)
    prediction.append(predicted)
    
tp,tn,fp,fn = 0,0,0,0
for doc in range(len(test_X)):
    for sentence in range(len(test_X[doc]['gold_labels'])):
        if prediction[doc][sentence] == 0:
            if test_X[doc]['gold_labels'][sentence] == 0:
                tp += 1
            else:
                fp += 1
        else:
            if test_X[doc]['gold_labels'][sentence] == 1:
                tn += 1
            else:
                fn += 1
num_of_sentence=[val for sublist in prediction for val in sublist]
print('Accuracy for training length '+str(train_percent)+'% : ',((tp+tn)/len(num_of_sentence))*100)