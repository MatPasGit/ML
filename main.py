from scipy.stats import ttest_ind
import numpy as np
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

#Pobranie zawartości zestawu danych

dataset = 'credit_risk/original.csv'
dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
#read = pd.read_csv("datasets/%s.csv" % (dataset))

X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)
print(X)
print(y)

#Walidacja krzyżowa (5 powtórzeń 2-krotnej walidacji krzyżowej)
n_splits = 2
n_repeats = 5

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

ensamble_methods = ["bagging", "adaboost", "random subspace"]
combination_methods = ["majority", "weighted", "Bordy"]
number_of_classificators = [5, 10, 15]
basic_classificators = ["SVM", "Decision tree"]













#BAGGING



#ADABOOST



#RANDOM SUBSPACE


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



#TODO:
#KOD MA BYĆ JEDNĄ PŁYNNĄ ŚCIANĄ TEKSTU, BEZ ODRĘBNYCH PLIKÓW, KLAS I FUNKCJI !!!!
# 1.zmiana zestawów danych na pojedyncze pliki, wszystkie rekordy muszą być kompletne!!!!, ostatnia kolumna to klasa
# 10. implementacja testów statystycznych t-Studenta
# 11. implementacja testów statystycznych Wilcoxona
#A na koniec wyniki, wykresy i testy statystyczne
