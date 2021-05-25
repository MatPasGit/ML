import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # svm.LinearSVC(dual=False), SVC()
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon
import pandas as pd

class EnsembleModel(BaseEnsemble, ClassifierMixin):

    def __init__(self, ensemble_method ="RandomSubspace" ,base_estimator=None, n_estimators=10, n_subspace_features=5, voting="majority", random_state=None):
        self.ensemble_method = ensemble_method
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_subspace_features = n_subspace_features
        self.voting = voting
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        #Random Subspace
        self.n_features = X.shape[1]
        if self.n_subspace_features > self.n_features:
            raise ValueError(
                "Number of features in subspace higher than number of features.")

        self.subspaces = np.random.randint(
            0, self.n_features, (self.n_estimators, self.n_subspace_features))

        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(
                clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        if self.voting == "majority":
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
            pred_ = np.array(pred_)
            prediction = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
            return self.classes_[prediction]
        elif self.voting == "vector":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]
        else: #TODO weighted
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)


#MAIN EXPERIMENT
dataset = 'credit_risk/original.csv'
dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

print("Total number of features", X.shape[1])

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

ensemble_methods = ["RandomSubspace"] # "Bagging", "Adaboost",
base_estimators = [GaussianNB(), DecisionTreeClassifier(), KNeighborsClassifier()]
number_of_classificators = [5, 10, 15]
combination_methods = ["majority",  "vector"]  # "weighted",
#pd.set_option("display.max_rows", None, "display.max_columns", None)

test_results = pd.DataFrame()
wilcoxonarray = []

for ensemble_method in ensemble_methods:
    for base in base_estimators:
        for number in number_of_classificators:
            for combination in combination_methods:

                clf = EnsembleModel(ensemble_method=ensemble_method, base_estimator=base, n_estimators=number, n_subspace_features=1, voting=combination, random_state=123)
                scores = []
                subwilcoxon = []
                first = True

                for train, test in rskf.split(X, y):
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])

                    if first:
                        subwilcoxon = y_pred.copy()
                        first = False
                    else:
                        subwilcoxon = np.add(subwilcoxon, y_pred)

                    scores.append([accuracy_score(y[test], y_pred), 
                                recall_score(y[test], y_pred),
                                precision_score(y[test], y_pred, average='weighted', labels=np.unique(y_pred))])
                print(f"{str(ensemble_method)}, {str(base)}, {str(number)}, {str(combination)}"+" = Accuracy score: %.3f (%.3f)" % (
                    np.mean(scores, axis=0)[0], np.std(scores, axis=0)[0]))

                test_results = test_results.append({
                    'Ensemble method': str(ensemble_method),
                    'Model': str(base),
                    'Number of classificators': str(number),
                    'Voting': str(combination),
                    'accuracy': round(np.mean(scores, axis=0)[0], 4),
                    'accuracy std': "("+str(round(np.std(scores, axis=0)[0], 4))+")",
                    'recall': round(np.mean(scores, axis=0)[1], 4),
                    'recall std': "("+str(round(np.std(scores, axis=0)[1], 4))+")",
                    'precision': round(np.mean(scores, axis=0)[2], 4),
                    'precision std': "("+str(round(np.std(scores, axis=0)[2], 4))+")"
                }, ignore_index=True)

                wilcoxonarray.append(subwilcoxon/len(subwilcoxon))


#Wilcoxon stats
wilcoxonStat = pd.DataFrame()

for arrayOne in range(len(wilcoxonarray)):
    for arraySecond in range(arrayOne+1):
        if arrayOne != arraySecond:
            if not(np.array_equal(wilcoxonarray[arrayOne], wilcoxonarray[arraySecond])):
                stat, p = wilcoxon(wilcoxonarray[arrayOne], wilcoxonarray[arraySecond])

                param_dict = {
                        '1. prediction - ensemble': test_results.get('Ensemble method')[arrayOne],
                        '2. prediction - ensemble': test_results.get('Ensemble method')[arraySecond],
                        '1. prediction - model': test_results.get('Model')[arrayOne],
                        '2. prediction - model': test_results.get('Model')[arraySecond],
                        '1. prediction - number of classificators': test_results.get('Number of classificators')[arrayOne],
                        '2. prediction - number of classificators': test_results.get('Number of classificators')[arraySecond],
                        '1. prediction - voting': test_results.get('Voting')[arrayOne],
                        '2. prediction - voting': test_results.get('Voting')[arraySecond],
                        'stat': stat,
                        'p': p,
                    }

                alpha = 0.05
                if p > alpha:
                    param_dict['rozkład'] = 'Taki sam'
                else:
                    param_dict['rozkład'] = 'Różny'
                wilcoxonStat = wilcoxonStat.append(param_dict, ignore_index=True)


#Print and save to csv files
test_results = test_results.sort_values(by=['accuracy'])
test_results_csv = test_results[['Ensemble method', 'Model', 'Number of classificators', 'Voting',
                                 'accuracy', 'accuracy std', 'recall', 'recall std', 'precision', 'precision std']]
print(test_results_csv)

print("Znalezione podobne dystrybuanty:")
wilcoxonStat = wilcoxonStat.sort_values(by=['p'])
wilcoxonStat_csv = wilcoxonStat[['1. prediction - ensemble', '2. prediction - ensemble', '1. prediction - model', '1. prediction - number of classificators',
                                 '1. prediction - voting', '2. prediction - model', '2. prediction - number of classificators',
                                 '2. prediction - voting', 'stat', 'p', 'rozkład']]
print(wilcoxonStat_csv)

test_results_csv.to_csv(r'.\test_results.csv', index = False, header=True)
wilcoxonStat_csv.to_csv(r'.\wilcoxonStat.csv', index = False, header=True)


#TODO: Bagging, boosting and ważone głosowanie
