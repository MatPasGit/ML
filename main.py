from tabulate import tabulate
from scipy.stats import ttest_ind
import numpy as np
from numpy import set_printoptions
from scipy import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, KFold


#Pobranie zawartości zestawu danych

dataset = 'credit_risk/original.csv'
dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
#read = pd.read_csv("datasets/%s.csv" % (dataset))

X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

#Walidacja krzyżowa (5 powtórzeń 2-krotnej walidacji krzyżowej)
n_splits = 2
n_repeats = 5

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

#Sieć neuronowa jako klasifykator
network = MLPClassifier(hidden_layer_sizes=200, momentum=0, random_state=1234)

scores=[]
firstScore = True
confusionMatrixMean = []

for train_index, test_index in rskf.split(X, y):
         x_train, x_test = X[train_index], X[test_index]
         y_train, y_test = y[train_index], y[test_index]

         network.fit(x_train, y_train)
         predict = network.predict(x_test)
         score = accuracy_score(y_test, predict)
         scores.append(score)            # Wypisanie wyniku

mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"\nSrednia accuracy: {mean_score}\nOdchylenie: {std_score}\n")



#TODO:
#KOD MA BYĆ JEDNĄ PŁYNNĄ ŚCIANĄ TEKSTU, BEZ ODRĘBNYCH PLIKÓW, KLAS I FUNKCJI !!!!
# 1.zmiana zestawów danych na pojedyncze pliki, wszystkie rekordy muszą być kompletne!!!!, ostatnia kolumna to klasa
# 2.implementacja 1. metody zespołowej - baggingu
# 3.implementacja 2. metody zespołowej - adaboostu
# 4.implementacja 3. metody zespołowej - random subspace
# 5.implementacja 1. metody kombinacji - głosowanie większościowe
# 6.implementacja 2. metody kombinacji - głosowanie ważone
# 7.implementacja 3. metody kombinacji - metoda Bordy

# Każda z prób jest homogeniczna, liczba klasyfikatorów w puli: 5, 10, 15
# 8. implementacja 1. klasyfikatora (albo bierzemy z biblioteki) - SVM
# 9. implementacja 2. klasyfikatora (albo bierzemy z biblioteki) - Drzewo decyzyjne

# 10. implementacja testów statystycznych t-Studenta
# 11. implementacja testów statystycznych Wilcoxona


#     #DECISION TREE

#     DDclassifier= DecisionTreeClassifier(criterion="entropy")
#     X = read.drop('default', axis = 1)
#     y = read['default']
#     np.where(y.values >= np.finfo(y.float64).max)
#     print(y)
#     DDclassifier.fit(X = X, y = y)

#     #SVM KLASYFIKATOR
#     SVMClassifier= svm.LinearSVC() ##Liniowy kernel Maszyny wektorów nośnych
#     SVMClassifier.fit(X,y)




#Przykładowa inicjalizacja zespołów klasyfikatorów
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=10, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=20, hard_voting=False, random_state=123)

# class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):
#     """
#     Random subspace ensemble
#     Komitet klasyfikatorow losowych podprzestrzeniach cech
#     """

#     def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=None):
#         # Klasyfikator bazowy
#         self.base_estimator = base_estimator
#         # Liczba klasyfikatorow
#         self.n_estimators = n_estimators
#         # Liczba cech w jednej podprzestrzeni
#         self.n_subspace_features = n_subspace_features
#         # Tryb podejmowania decyzji
#         self.hard_voting = hard_voting
#         # Ustawianie ziarna losowosci
#         self.random_state = random_state
#         np.random.seed(self.random_state)


#     def fit(self, X, y):
#         # Sprawdzenie czy X i y maja wlasciwy ksztalt
#         X, y = check_X_y(X, y)
#         # Przehowywanie nazw klas
#         self.classes_ = np.unique(y)

#         # Zapis liczby atrybutow
#         self.n_features = X.shape[1]
#         # Czy liczba cech w podprzestrzeni jest mniejsza od calkowitej liczby cech
#         if self.n_subspace_features > self.n_features:
#             raise ValueError(
#                 "Number of features in subspace higher than number of features.")
#         # Wylosowanie podprzestrzeni cech
#         self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))

#         # Wyuczenie nowych modeli i stworzenie zespolu
#         self.ensemble_ = []
#         for i in range(self.n_estimators):
#             self.ensemble_.append(
#                 clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

#         return self

#         def predict(self, X):
#             # Sprawdzenie czy modele sa wyuczone
#             check_is_fitted(self, "classes_")
#             # Sprawdzenie poprawnosci danych
#             X = check_array(X)
#             # Sprawdzenie czy liczba cech się zgadza
#             if X.shape[1] != self.n_features:
#                 raise ValueError("number of features does not match")
#             if self.hard_voting:
#                 # Podejmowanie decyzji na podstawie twardego glosowania
#                 pred_ = []
#                 # Modele w zespole dokonuja predykcji
#                 for i, member_clf in enumerate(self.ensemble_):
#                     pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
#                 # Zamiana na miacierz numpy (ndarray)
#                 pred_ = np.array(pred_)
#                 # Liczenie glosow
#                 prediction = np.apply_along_axis(
#                     lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
#                 # Zwrocenie predykcji calego zespolu
#                 return self.classes_[prediction]
#             else:
#                 # Podejmowanie decyzji na podstawie wektorow wsparcia
#                 esm = self.ensemble_support_matrix(X)
#                 # Wyliczenie sredniej wartosci wsparcia
#                 average_support = np.mean(esm, axis=0)
#                 # Wskazanie etykiet
#                 prediction = np.argmax(average_support, axis=1)
#                 # Zwrocenie predykcji calego zespolu
#                 return self.classes_[prediction]

#         def ensemble_support_matrix(self, X):
#             # Wyliczenie macierzy wsparcia
#             probas_ = []
#             for i, member_clf in enumerate(self.ensemble_):
#                 probas_.append(member_clf.predict_proba(
#                     X[:, self.subspaces[i]]))
#             return np.array(probas_)
