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
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_score

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
classificators_weights = {
    5: [1,2,3,2,1],
    10: [1,2,3,2,1,1,2,3,2,1],
    15: [1,2,3,2,1,1,2,3,2,1,1,2,3,2,1]}

classifier = None
method = None
number = 0
combination = None
for ensamble_method in ensamble_methods:
    for combination_method in combination_methods:
        for number_of_classificator in number_of_classificators:
            for basic_classificator in basic_classificators:

                if basic_classificator == "SVM":
                    # Liniowy kernel Maszyny wektorów nośnych; wybieramy LinearSVC ponieważ dużo próbek i więcej niż 2 klasy w niektórych przypadkach; decision_function_shape='ovo'
                    classifier = svm.LinearSVC(dual=False)
                    #classifier = DecisionTreeClassifier()
                else:
                    classifier = DecisionTreeClassifier()

                if combination_method == "majority":
                    combination = VotingClassifier(estimators=classifier)
                if combination_method == "weighted":
                    combination = VotingClassifier(estimators=classifier, weights=classificators_weights[number_of_classificator])
                if combination_method == "Bordy":
                    combination = VotingClassifier(estimators=classifier, voting='soft')

                if ensamble_method == "bagging":
                    method = BaggingClassifier(base_estimator=classifier, n_estimators=number_of_classificator, random_state=1234)
                if ensamble_method == "adaboost":
                    method = AdaBoostClassifier(base_estimator=classifier, n_estimators=number_of_classificator, random_state=1234, algorithm='SAMME')
                if ensamble_method == "random subspace":
                    method = BaggingClassifier(base_estimator=classifier, n_estimators=number_of_classificator, bootstrap=False, random_state=1234)

                scores = cross_val_score(method, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)

                print(f"\n{ensamble_method}, {combination_method}, {number_of_classificator}, {basic_classificator}")
                print(f"\nSrednia accuracy: {np.mean(scores)}\nOdchylenie: {np.std(scores)}\n")





                #Na początek inicjalizacja klasyfikatora w podanej liczbie

                #Potem zastowowanie metody zespołowej przy dostarczaniu danych

                #Klasyfikatory mielą

                #Decydowanie o ostatecznym wyniku działania funkcji (metoda kombinacji)

#A na koniec wyniki, wykresy i testy statystyczne




#TODO:
#KOD MA BYĆ JEDNĄ PŁYNNĄ ŚCIANĄ TEKSTU, BEZ ODRĘBNYCH PLIKÓW, KLAS I FUNKCJI !!!!
# 1.zmiana zestawów danych na pojedyncze pliki, wszystkie rekordy muszą być kompletne!!!!, ostatnia kolumna to klasa
# 10. implementacja testów statystycznych t-Studenta
# 11. implementacja testów statystycznych Wilcoxona





#Przykładowa inicjalizacja zespołów klasyfikatorów
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=10, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=False, random_state=123)
# clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=20, hard_voting=False, random_state=123)

class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):
    """
    Random subspace ensemble
    Komitet klasyfikatorow losowych podprzestrzeniach cech
    """

    def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=None):
        # Klasyfikator bazowy
        self.base_estimator = base_estimator
        # Liczba klasyfikatorow
        self.n_estimators = n_estimators
        # Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        # Tryb podejmowania decyzji
        self.hard_voting = hard_voting
        # Ustawianie ziarna losowosci
        self.random_state = random_state
        np.random.seed(self.random_state)


    def fit(self, X, y):
        # Sprawdzenie czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # Przehowywanie nazw klas
        self.classes_ = np.unique(y)

        # Zapis liczby atrybutow
        self.n_features = X.shape[1]
        # Czy liczba cech w podprzestrzeni jest mniejsza od calkowitej liczby cech
        if self.n_subspace_features > self.n_features:
            raise ValueError(
                "Number of features in subspace higher than number of features.")
        # Wylosowanie podprzestrzeni cech
        self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))

        # Wyuczenie nowych modeli i stworzenie zespolu
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(
                clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

        return self

        def predict(self, X):
            # Sprawdzenie czy modele sa wyuczone
            check_is_fitted(self, "classes_")
            # Sprawdzenie poprawnosci danych
            X = check_array(X)
            # Sprawdzenie czy liczba cech się zgadza
            if X.shape[1] != self.n_features:
                raise ValueError("number of features does not match")
            if self.hard_voting:
                # Podejmowanie decyzji na podstawie twardego glosowania
                pred_ = []
                # Modele w zespole dokonuja predykcji
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
                # Zamiana na miacierz numpy (ndarray)
                pred_ = np.array(pred_)
                # Liczenie glosow
                prediction = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
                # Zwrocenie predykcji calego zespolu
                return self.classes_[prediction]
            else:
                # Podejmowanie decyzji na podstawie wektorow wsparcia
                esm = self.ensemble_support_matrix(X)
                # Wyliczenie sredniej wartosci wsparcia
                average_support = np.mean(esm, axis=0)
                # Wskazanie etykiet
                prediction = np.argmax(average_support, axis=1)
                # Zwrocenie predykcji calego zespolu
                return self.classes_[prediction]

        def ensemble_support_matrix(self, X):
            # Wyliczenie macierzy wsparcia
            probas_ = []
            for i, member_clf in enumerate(self.ensemble_):
                probas_.append(member_clf.predict_proba(
                    X[:, self.subspaces[i]]))
            return np.array(probas_)
