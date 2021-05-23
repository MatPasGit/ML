from scipy.stats import ttest_ind, wilcoxon
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

from sklearn.svm import SVC

#Pobranie zawartości zestawu danych


#read = pd.read_csv("datasets/%s.csv" % (dataset))
# Dla DecisionTree
bagging_acc_list_majority = list()
bagging_acc_list_weighted = list()
bagging_acc_list_bordy = list()

adaboost_acc_list_majority = list()
adaboost_acc_list_weighted = list()
adaboost_acc_list_bordy = list()

rf_acc_list_majority = list()
rf_acc_list_weighted = list()
rf_acc_list_bordy = list()

combination_methods = ["majority", "weighted", "Bordy"]
number_of_classificators = [5, 10, 15]
basic_classificators = ["SVM", "Decision tree"]

datasets = ['credit_risk/original.csv','1/heart.csv','bancrupcy/train.csv', 'contaceptive/contraceptive.dat','bancrupcy/train.csv', 'employee-attrition/employee_attrition_train.csv',"datasets/employee-satisfaction/Employee Satisfaction Index.csv",
'datasets/fakenews/FA-KES-Dataset.csv' ,'datasets/income/income_evaluation.csv',
'datasets/MobilePriceClassification/train.csv','datasets/mushrooms/mushrooms.csv','datasets/nursery/nursery.dat',
'datasets/rain in austraila/weatherAUS.csv','datasets/realEstate/data.csv','datasets/serverlogs/CIDDS-001-external-week1.csv'
,'datasets/stroke/healthcare-dataset-stroke-data.csv', 'datasets/telescope/magic.dat','datasets/titanic/full.csv'
,'datasets/usstocks/2014_Financial_Data.csv', 'datasets/usstocks/2015_Financial_Data.csv',
'datasets/waterqual/water_potability.csv' ,'datasets/wine/winequality-red.csv']
for d in datasets:

    dataset = 'credit_risk/original.csv'
    dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
    # read = pd.read_csv("datasets/%s.csv" % (dataset))
    dataset[~np.isnan(dataset).any(axis=1), :]
    X = dataset[1:, :-1]
    y = dataset[1:, -1].astype(int)
    print(X)
    print(y)

    #Walidacja krzyżowa (5 powtórzeń 2-krotnej walidacji krzyżowej)
    #n_splits = 2
    #n_repeats = 5

    #rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

    #ensamble_methods = ["bagging", "adaboost", "random subspace"]

    #BAGGING

    print("BAGGING\n")

    for combination_method in combination_methods:
        for number_of_classificator in number_of_classificators:
            for basic_classificator in basic_classificators:

                if basic_classificator == "Decision tree":
                    print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    for i in range(number_of_classificator):
                            classificatorsList.append(('classificator no.'+str(i+1), DecisionTreeClassifier()))
                    #print(classificatorsList)

                    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1234)

                    scores = list()

                    for i in range(number_of_classificator):

                        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList[i][1].fit(X_train, y_train)
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val)
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

                    #print(scores)

                    if combination_method == "majority":
                        #ensemble = VotingClassifier(estimators=classificatorsList)
                        predictionArray = []
                        for i in range(number_of_classificator):
                            predictionArray.append(classificatorsList[i][1].predict(X_test))
                        prediction = np.array(predictionArray)
                        prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=prediction.T)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                            bagging_acc_list_majority.append(score)

                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)

                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                            bagging_acc_list_weighted.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy": #Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        predictionArray = []
                        for i in range(number_of_classificator):
                            predictionArray.append(classificatorsList[i][1].predict_proba(X_test))
                        esm = np.array(predictionArray)
                        # Wyliczenie sredniej wartosci wsparcia
                        average_support = np.mean(esm, axis=0)
                        # Wskazanie etykiet
                        prediction = np.argmax(average_support, axis=1)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator =="Decision tree":
                            bagging_acc_list_bordy.append(score)

                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                elif basic_classificator == "SVM":

                    print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    for i in range(number_of_classificator):
                        classificatorsList.append(
                            ('classificator no.'+str(i+1), svm.LinearSVC(dual=False)))
                    #print(classificatorsList)

                    X_train_full, X_test, y_train_full, y_test = train_test_split(
                        X, y, test_size=0.50, random_state=1234)

                    scores = list()

                    for i in range(number_of_classificator):

                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList[i][1].fit(X_train, y_train)
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val)
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

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
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy":  # Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        print("Brak metody z uwagi na brak odpowiedniej funkcji")


    #ADABOOST

    print("ADABOOST\n")

    for combination_method in combination_methods:
        for number_of_classificator in number_of_classificators:
            for basic_classificator in basic_classificators:

                if basic_classificator == "Decision tree":
                    print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    for i in range(number_of_classificator):
                        classificatorsList.append(('classificator no.'+str(i+1), DecisionTreeClassifier()))
                    #print(classificatorsList)

                    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1234)

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
                            X_train_part, y_train_part, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList[i][1].fit(X_train, y_train)
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val)
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

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
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           adaboost_acc_list_majority.append(score)

                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           adaboost_acc_list_weighted.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy": #Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        predictionArray = []
                        for i in range(number_of_classificator):
                            predictionArray.append(classificatorsList[i][1].predict_proba(X_test))
                        esm = np.array(predictionArray)
                        # Wyliczenie sredniej wartosci wsparcia
                        average_support = np.mean(esm, axis=0)
                        # Wskazanie etykiet
                        prediction = np.argmax(average_support, axis=1)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           adaboost_acc_list_bordy.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                elif basic_classificator == "SVM":

                    print(
                        f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    for i in range(number_of_classificator):
                        classificatorsList.append(
                            ('classificator no.'+str(i+1), svm.LinearSVC(dual=False)))
                    #print(classificatorsList)

                    X_train_full, X_test, y_train_full, y_test = train_test_split(
                        X, y, test_size=0.50, random_state=1234)

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
                                #X_train_part = np.concatenate(X_train_part, X_train_split[j])
                                #y_train_part = np.concatenate(y_train_part, y_train_split[j])
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_part, y_train_part, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList[i][1].fit(X_train, y_train)
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val)
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

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
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy":  # Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        print("Brak metody z uwagi na brak odpowiedniej funkcji")


    #RANDOM SUBSPACE

    print("RANDOM SUBSPACE\n")

    for combination_method in combination_methods:
        for number_of_classificator in number_of_classificators:
            for basic_classificator in basic_classificators:

                if basic_classificator == "Decision tree":
                    print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.50, random_state=1234)

                    scores = list()

                    n_features = X.shape[1]
                    n_subspace_features = math.floor(0.5*X.shape[1])

                    subspaces = np.random.randint(0, n_features, (number_of_classificator, n_subspace_features))

                    for i in range(number_of_classificator):
                        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList.append(('classificator no.'+str(i+1), clone(DecisionTreeClassifier()).fit(X_train[:, subspaces[i]], y_train)))
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val[:, subspaces[i]])
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

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
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           rf_acc_list_majority.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           rf_acc_list_weighted.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy":  # Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        predictionArray = []
                        for i in range(number_of_classificator):
                            predictionArray.append(
                                classificatorsList[i][1].predict_proba(X_test[:, subspaces[i]]))
                        esm = np.array(predictionArray)
                        # Wyliczenie sredniej wartosci wsparcia
                        average_support = np.mean(esm, axis=0)
                        # Wskazanie etykiet
                        prediction = np.argmax(average_support, axis=1)
                        score = accuracy_score(y_test, prediction)
                        if number_of_classificator == 10 and basic_classificator == "Decision tree":
                           rf_acc_list_bordy.append(score)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                elif basic_classificator == "SVM":

                    print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                    classificatorsList = list()

                    X_train_full, X_test, y_train_full, y_test = train_test_split(
                        X, y, test_size=0.50, random_state=1234)

                    scores = list()

                    n_features = X.shape[1]
                    n_subspace_features = math.floor(0.5*X.shape[1])

                    subspaces = np.random.randint(
                        0, n_features, (number_of_classificator, n_subspace_features))

                    for i in range(number_of_classificator):
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                        # fit the model
                        classificatorsList.append(('classificator no.'+str(i+1), clone(
                            svm.LinearSVC(dual=False)).fit(X_train[:, subspaces[i]], y_train)))
                        # evaluate the model
                        yhat = classificatorsList[i][1].predict(X_val[:, subspaces[i]])
                        acc = accuracy_score(y_val, yhat)
                        # store the performance
                        scores.append(acc)
                        # report model performance

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
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

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
                                k[1] = 0
                            prediction.append(maxclass)
                        score = accuracy_score(y_test, prediction)
                        print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                    if combination_method == "Bordy":  # Głosowanie miękkie za pomoca wektorów wsparcia
                        #ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
                        print("Brak metody z uwagi na brak odpowiedniej funkcji")




stat,p = wilcoxon(bagging_acc_list_majority,adaboost_acc_list_majority )
print("bagging vs adaboost majority voting wilcoxon test")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ",stat )

stat,p = wilcoxon(bagging_acc_list_weighted,adaboost_acc_list_weighted )
print("bagging vs adaboost weighted voting wilcoxon test")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ",stat )
stat,p = wilcoxon(bagging_acc_list_bordy,adaboost_acc_list_bordy )
print("bagging vs adaboost bordy wilcoxon test")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ",stat )
stat,p = wilcoxon(bagging_acc_list_majority,rf_acc_list_majority )
print("bagging vs random subs wilcoxon test")
print("majority voting")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ", stat)
stat,p = wilcoxon(bagging_acc_list_weighted,rf_acc_list_weighted )
print("weighted")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ", stat)
stat,p = wilcoxon(bagging_acc_list_bordy,rf_acc_list_bordy )
print("bordy")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ",stat )
stat,p = wilcoxon(adaboost_acc_list_majority,rf_acc_list_majority )
print("adaboost vs random subs wilcoxon test")
print("majority voting")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ", stat)
stat,p = wilcoxon(adaboost_acc_list_weighted,rf_acc_list_weighted )
print("weighted:")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ", stat)
stat,p = wilcoxon(adaboost_acc_list_bordy,rf_acc_list_bordy )
print("bordy:")
if p > 0.05:
	print('Same distribution ')
else:
	print('Different distribution ')
print("Stat: ",stat )



# classificatorsList = list()

# for i in range(3):
#     classificatorsList.append(
#         ('classificator no.'+str(i+1), DecisionTreeClassifier()))

# print(classificatorsList)

# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X, y, test_size=0.50, random_state=1)

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full, y_train_full, test_size=0.33, random_state=1)

# scores = list()

# for name, model in classificatorsList:

# 	# fit the model
# 	model.fit(X_train, y_train)
# 	# evaluate the model
# 	yhat = model.predict(X_val)
# 	acc = accuracy_score(y_val, yhat)
# 	# store the performance
# 	scores.append(acc)
# 	# report model performance

# print(scores)

# #evaluate_models(models, X_train, X_val, y_train, y_val):

# ensemble = VotingClassifier(
#     estimators=classificatorsList, voting='soft', weights=scores)
# # fit the ensemble on the training dataset
# ensemble.fit(X_train_full, y_train_full)
# # make predictions on test set
# yhat = ensemble.predict(X_test)
# # evaluate predictions
# score = accuracy_score(y_test, yhat)
# print('Weighted Avg Accuracy: %.3f' % (score*100))
# # evaluate each standalone model

# for name, model in classificatorsList:
# 	# fit the model
# 	model.fit(X_train_full, y_train_full)
# 	# evaluate the model
# 	yhat = model.predict(X_test)
# 	acc = accuracy_score(y_test, yhat)
# 	# store the performance
# 	scores.append(acc)
# 	# report model performance


# for i in range(len(classificatorsList)):
# 	print('>%s: %.3f' % (classificatorsList[i][0], scores[i]*100))
# # evaluate equal weighting
# ensemble = VotingClassifier(estimators=classificatorsList, voting='soft')
# ensemble.fit(X_train_full, y_train_full)
# yhat = ensemble.predict(X_test)
# score = accuracy_score(y_test, yhat)
# print('Voting Accuracy: %.3f' % (score*100))


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

# if combination_method == "Bordy":
                #     print("Operacja nie jest możliwa")

                # else:
                #     print(f"{combination_method}, {number_of_classificator}, {basic_classificator}")

                #     classificatorsList = list()

                #     for i in range(number_of_classificator):
                #         classificatorsList.append(('classificator no.'+str(i+1), svm.LinearSVC(dual=False)))
                #     #print(classificatorsList)

                #     X_train_full, X_test, y_train_full, y_test = train_test_split(
                #         X, y, test_size=0.50, random_state=1234)

                #     scores = list()

                #     for i in range(number_of_classificator):
                #         X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.50, random_state=1234+i)
                #         classificatorsList[i][1].fit(X_train, y_train)
                #         yhat = classificatorsList[i][1].predict(X_val)
                #         acc = accuracy_score(y_val, yhat)
                #         scores.append(acc)

                #     #print(scores)

                #     if combination_method == "majority":
                #         predictionArray = []
                #         for i in range(number_of_classificator):
                #             predictionArray.append(classificatorsList[i][1].predict(X_test))
                #         prediction = np.array(predictionArray)
                #         prediction = np.apply_along_axis(lambda x: np.argmax(
                #             np.bincount(x)), axis=1, arr=prediction.T)
                #         score = accuracy_score(y_test, prediction)
                #         print('Weighted Avg Accuracy: %.3f\n' % (score*100))

                #     if combination_method == "weighted":
                #         ensemble = VotingClassifier(estimators=classificatorsList, weights=scores)








#TODO:
#KOD MA BYĆ JEDNĄ PŁYNNĄ ŚCIANĄ TEKSTU, BEZ ODRĘBNYCH PLIKÓW, KLAS I FUNKCJI !!!!
# 1.zmiana zestawów danych na pojedyncze pliki, wszystkie rekordy muszą być kompletne!!!!, ostatnia kolumna to klasa
# 10. implementacja testów statystycznych t-Studenta
# 11. implementacja testów statystycznych Wilcoxona
#A na koniec wyniki, wykresy i testy statystyczne
#Posprzątać lekko kod
#Zaimplementować walidację krzyżową