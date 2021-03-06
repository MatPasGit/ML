from scipy.stats import ttest_ind
import numpy as np
import math
from numpy import mean
from numpy import std
from numpy import set_printoptions
from scipy import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_score, train_test_split
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from scipy.stats import wilcoxon

#Pobranie zawartości zestawu danych

dataset = 'credit_risk/original.csv'
dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
#read = pd.read_csv("datasets/%s.csv" % (dataset))

X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)
#print(X)
#print(y)

wilcoxonarray = []

combination_methods = ["majority", "weighted", "Bordy"]
number_of_classificators = [5, 10, 15]
basic_classificators = ["SVM", "Decision tree"]

#BAGGING

print("BAGGING\n")

for combination_method in combination_methods:
    for number_of_classificator in number_of_classificators:
        for basic_classificator in basic_classificators:

            basicscores = []
            subwilcoxon = []
            for cross_time in range(5):
                for cross_page in range(2):

                    if basic_classificator == "Decision tree":
                        
                        classificatorsList = list()

                        for i in range(number_of_classificator):                    
                                classificatorsList.append(('classificator no.'+str(i+1), DecisionTreeClassifier()))
                        #print(classificatorsList)
                       
                        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1234+cross_time)
                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp

                        scores = list()

                        for i in range(number_of_classificator):

                            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                            classificatorsList[i][1].fit(X_train, y_train)
                            yhat = classificatorsList[i][1].predict(X_val)
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)

                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(classificatorsList[i][1].predict(X_test))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(classificatorsList[i][1].predict(X_test))
                            
                            weightArray = []
                            for i in set(y):
                                weightArray.append([i,0])

                            prediction=[]
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "Bordy": #Głosowanie miękkie za pomoca wektorów wsparcia
                            #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(classificatorsList[i][1].predict_proba(X_test))
                            esm = np.array(predictionArray)
                            average_support = np.mean(esm, axis=0)
                            prediction = np.argmax(average_support, axis=1)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                    elif basic_classificator == "SVM":

                        classificatorsList = list()

                        for i in range(number_of_classificator):
                            classificatorsList.append(
                                ('classificator no.'+str(i+1), svm.LinearSVC(dual=False)))
                        #print(classificatorsList)

                        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1234+cross_time)
                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp

                        scores = list()

                        for i in range(number_of_classificator):

                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_full, y_train_full, test_size=0.50, random_state=1234+i+cross_time*20)
                            classificatorsList[i][1].fit(X_train, y_train)
                            yhat = classificatorsList[i][1].predict(X_val)
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)

                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(
                                np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test))

                            weightArray = []
                            for i in set(y):
                                weightArray.append([i, 0])

                            prediction = []
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

            print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

            if ((basic_classificator == "SVM") and (combination_method == "Bordy")):
                print("Brak metody z uwagi na brak odpowiedniej funkcji")
            else:
                print("Accuracy:" + str(np.mean(basicscores))+", std: "+str(np.std(basicscores)))
                wilcoxon = []
                for i in range(len(subwilcoxon)):
                    if i == 0:
                        wilcoxon = subwilcoxon[i].copy()
                    else:
                        wilcoxon = np.add(wilcoxon,subwilcoxon[i])
                        #wilcoxon = np.concatenate([wilcoxon,subwilcoxon[i]])
                wilcoxonarray.append(wilcoxon/len(wilcoxon))

            

# for j in range(3):
#     for i in range(15):
#         if i == 0:
#             wilcoxonarraymean.append(wilcoxonarray[j*15+i])
#         else:
#             wilcoxonarraymean[j] = np.concatenate([wilcoxonarraymean[j],wilcoxonarray[j*15+i]])

#ADABOOST

print("ADABOOST\n")

for combination_method in combination_methods:
    for number_of_classificator in number_of_classificators:
        for basic_classificator in basic_classificators:

            basicscores = []
            subwilcoxon = []
            for cross_time in range(5):
                for cross_page in range(2):

                    if basic_classificator == "Decision tree":

                        classificatorsList = list()

                        for i in range(number_of_classificator):
                            classificatorsList.append(('classificator no.'+str(i+1), DecisionTreeClassifier()))
                        #print(classificatorsList)

                        X_train_full, X_test, y_train_full, y_test = train_test_split(
                            X, y, test_size=0.50, random_state=1234)

                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp
                        
                        scores = list()

                        X_train_split = np.array_split(
                            X_train_full, number_of_classificator)
                        y_train_split = np.array_split(
                            y_train_full, number_of_classificator)

                        for i in range(number_of_classificator):
                            X_train_part = X_train_split[0]
                            y_train_part = y_train_split[0]
                            if i > 0:
                                for j in range(1,i+1):
                                    for k in X_train_split[j]:
                                        X_train_part += k
                                    for k in y_train_split[j]:
                                        y_train_part += k
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_part, y_train_part, test_size=0.50, random_state=1234)
                            classificatorsList[i][1].fit(X_train, y_train)
                            yhat = classificatorsList[i][1].predict(X_val)
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)
                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(
                                np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(classificatorsList[i][1].predict(X_test))
                            
                            weightArray = []
                            for i in set(y):
                                weightArray.append([i,0])

                            prediction=[]
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "Bordy": #Głosowanie miękkie za pomoca wektorów wsparcia
                            #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(classificatorsList[i][1].predict_proba(X_test))
                            esm = np.array(predictionArray)
                            average_support = np.mean(esm, axis=0)
                            prediction = np.argmax(average_support, axis=1)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                    elif basic_classificator == "SVM":

                        classificatorsList = list()

                        for i in range(number_of_classificator):
                            classificatorsList.append(
                                ('classificator no.'+str(i+1), svm.LinearSVC(dual=False)))
                        #print(classificatorsList)

                        X_train_full, X_test, y_train_full, y_test = train_test_split(
                            X, y, test_size=0.50, random_state=1234)

                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp

                        scores = list()

                        X_train_split = np.array_split(
                            X_train_full, number_of_classificator)
                        y_train_split = np.array_split(
                            y_train_full, number_of_classificator)

                        for i in range(number_of_classificator):
                            X_train_part = X_train_split[0]
                            y_train_part = y_train_split[0]
                            if i > 0:
                                for j in range(1,i+1):
                                    for k in X_train_split[j]:
                                        X_train_part += k
                                    for k in y_train_split[j]:
                                        y_train_part += k
                            
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_part, y_train_part, test_size=0.50, random_state=1234+i+cross_time*20)
                            classificatorsList[i][1].fit(X_train, y_train)
                            yhat = classificatorsList[i][1].predict(X_val)
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)

                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(
                                np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test))

                            weightArray = []
                            for i in set(y):
                                weightArray.append([i, 0])

                            prediction = []
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

            print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

            if ((basic_classificator == "SVM") and (combination_method == "Bordy")):
                print("Brak metody z uwagi na brak odpowiedniej funkcji")
            else:
                print("Accuracy:" + str(np.mean(basicscores))+", std: "+str(np.std(basicscores)))
                wilcoxon = []
                for i in range(len(subwilcoxon)):
                    if i == 0:
                        wilcoxon = subwilcoxon[i].copy()
                    else:
                        wilcoxon = np.add(wilcoxon, subwilcoxon[i])
                        #wilcoxon = np.concatenate([wilcoxon,subwilcoxon[i]])
                wilcoxonarray.append(wilcoxon/len(wilcoxon))
            

#RANDOM SUBSPACE

print("RANDOM SUBSPACE\n")

for combination_method in combination_methods:
    for number_of_classificator in number_of_classificators:
        for basic_classificator in basic_classificators:

            basicscores = []
            subwilcoxon = []
            for cross_time in range(5):
                for cross_page in range(2):

                    if basic_classificator == "Decision tree":

                        classificatorsList = list()

                        X_train_full, X_test, y_train_full, y_test = train_test_split(
                            X, y, test_size=0.50, random_state=1234+cross_time)

                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp

                        scores = list()

                        n_features = X.shape[1]
                        n_subspace_features = math.floor(0.5*X.shape[1])

                        subspaces = np.random.randint(0, n_features, (number_of_classificator, n_subspace_features))

                        for i in range(number_of_classificator):
                            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.50, random_state=1234+i+cross_time*20)
                            classificatorsList.append(('classificator no.'+str(i+1), clone(DecisionTreeClassifier()).fit(X_train[:, subspaces[i]], y_train)))
                            yhat = classificatorsList[i][1].predict(X_val[:, subspaces[i]])
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)

                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test[:, subspaces[i]]))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(
                                np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test[:, subspaces[i]]))

                            weightArray = []
                            for i in set(y):
                                weightArray.append([i, 0])

                            prediction = []
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "Bordy":  # Głosowanie miękkie za pomoca wektorów wsparcia
                            #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict_proba(X_test[:, subspaces[i]]))
                            esm = np.array(predictionArray)
                            average_support = np.mean(esm, axis=0)
                            prediction = np.argmax(average_support, axis=1)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                    elif basic_classificator == "SVM":


                        classificatorsList = list()

                        X_train_full, X_test, y_train_full, y_test = train_test_split(
                            X, y, test_size=0.50, random_state=1234+cross_time)

                        if cross_page == 1:
                            temp = X_train_full
                            X_train_full = X_test
                            X_test = temp

                            temp = y_train_full
                            y_train_full = y_test
                            y_test = temp

                        scores = list()

                        n_features = X.shape[1]
                        n_subspace_features = math.floor(0.5*X.shape[1])

                        subspaces = np.random.randint(
                            0, n_features, (number_of_classificator, n_subspace_features))

                        for i in range(number_of_classificator):
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_full, y_train_full, test_size=0.50, random_state=1234+i+cross_time*20)
                            classificatorsList.append(('classificator no.'+str(i+1), clone(
                                svm.LinearSVC(dual=False)).fit(X_train[:, subspaces[i]], y_train)))
                            yhat = classificatorsList[i][1].predict(X_val[:, subspaces[i]])
                            acc = accuracy_score(y_val, yhat)
                            scores.append(acc)

                        #print(scores)

                        if combination_method == "majority":
                            #ensemble = VotingClassifier(estimators=classificatorsList)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test[:, subspaces[i]]))
                            prediction = np.array(predictionArray)
                            prediction = np.apply_along_axis(lambda x: np.argmax(
                                np.bincount(x)), axis=1, arr=prediction.T)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)

                        if combination_method == "weighted":
                            #ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)
                            predictionArray = []
                            for i in range(number_of_classificator):
                                predictionArray.append(
                                    classificatorsList[i][1].predict(X_test[:, subspaces[i]]))

                            weightArray = []
                            for i in set(y):
                                weightArray.append([i, 0])

                            prediction = []
                            for j in range(len(predictionArray[0])):
                                for i in range(len(predictionArray)):
                                    for k in weightArray:
                                        if (k[0] == predictionArray[i][j]):
                                            k[1] += scores[i]
                                maxvalue = 0
                                maxclass = 0
                                for k in weightArray:
                                    if k[1] > maxvalue:
                                        maxclass = k[0]
                                        maxvalue = k[1]
                                    k[1] = 0
                                prediction.append(maxclass)
                            score = accuracy_score(y_test, prediction)
                            basicscores.append(score)
                            subwilcoxon.append(prediction)
        
            print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

            if ((basic_classificator == "SVM") and (combination_method == "Bordy")):
                print("Brak metody z uwagi na brak odpowiedniej funkcji")
            else:
                print("Accuracy:" + str(np.mean(basicscores))+", std: "+str(np.std(basicscores)))
                wilcoxon = []
                for i in range(len(subwilcoxon)):
                    if i == 0:
                        wilcoxon = subwilcoxon[i].copy()
                    else:
                        wilcoxon = np.add(wilcoxon,subwilcoxon[i])
                        #wilcoxon = np.concatenate([wilcoxon,subwilcoxon[i]])
                wilcoxonarray.append(wilcoxon/len(wilcoxon))
            


wilcoxonStat = pd.DataFrame()

for arrayOne in range(len(wilcoxonarray)):
    for arraySecond in range(arrayOne+1):
        if arrayOne != arraySecond:
            stat, p = wilcoxon(wilcoxonarray[arrayOne], wilcoxonarray[arraySecond])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                param_dict = {
                    'arrayOne': arrayOne,
                    'arratSecond': arraySecond,
                    'stat': stat,
                    'p': p,
                    'wynik': "="
                }
                wilcoxonStat = wilcoxonStat.append(param_dict, ignore_index=True)
            else:
                param_dict = {
                    'arrayOne': arrayOne,
                    'arratSecond': arraySecond,
                    'stat': stat,
                    'p': p,
                    'wynik': "!"
                }
                wilcoxonStat = wilcoxonStat.append(param_dict, ignore_index=True)
        else:
            param_dict = {
                'arrayOne': arrayOne,
                'arratSecond': arraySecond,
                'stat': "-",
                'p': "-",
                'wynik': "-"
            }
            wilcoxonStat = wilcoxonStat.append(param_dict, ignore_index=True)

print(wilcoxonStat)

















# classifier = None
# method = None
# number = 0
# combination = None
# for ensamble_method in ensamble_methods:
#     for combination_method in combination_methods:
#         for number_of_classificator in number_of_classificators:
#             for basic_classificator in basic_classificators:

#                 if basic_classificator == "SVM":
#                     # Liniowy kernel Maszyny wektorów nośnych; wybieramy LinearSVC ponieważ dużo próbek i więcej niż 2 klasy w niektórych przypadkach; decision_function_shape='ovo'
#                     classifier = svm.LinearSVC(dual=False)
#                     #classifier = DecisionTreeClassifier()
#                 else:
#                     classifier = DecisionTreeClassifier()

#                 if combination_method == "majority":
#                     combination = VotingClassifier(estimators=classifier)
#                 if combination_method == "weighted":
#                     combination = VotingClassifier(estimators=classifier, weights=classificators_weights[number_of_classificator])
#                 if combination_method == "Bordy":
#                     combination = VotingClassifier(estimators=classifier, voting='soft')

#                 if ensamble_method == "bagging":
#                     method = BaggingClassifier(base_estimator=classifier, n_estimators=number_of_classificator, random_state=1234)
#                 if ensamble_method == "adaboost":
#                     method = AdaBoostClassifier(base_estimator=classifier, n_estimators=number_of_classificator, random_state=1234, algorithm='SAMME')
#                 if ensamble_method == "random subspace":
#                     method = BaggingClassifier(base_estimator=classifier, n_estimators=number_of_classificator, bootstrap=False, random_state=1234)

#                 scores = cross_val_score(method, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)

#                 print(f"\n{ensamble_method}, {combination_method}, {number_of_classificator}, {basic_classificator}")
#                 print(f"\nSrednia accuracy: {np.mean(scores)}\nOdchylenie: {np.std(scores)}\n")
