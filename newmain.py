import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.stats import rankdata
import math

class BaggingEnsembleModel(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, voting="majority", random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        self.classificators_weights = np.ones((n_estimators,), dtype=int)
        np.random.seed(self.random_state)

    def fitmain(self, X, y):
        #Ustalenie wag klasyfikatorów w przypadku głosownaia ważonego
        if self.voting == "weighted":
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.50, random_state=1234)
            self.fit(X_train, y_train)
            self.predictweight(X_val, y_val)
        self.fit(X, y)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        self.ensemble_ = []
        for i in range(self.n_estimators):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=1234+i)
            self.ensemble_.append(clone(self.base_estimator).fit(X_train, y_train))

        return self

    def predictweight(self, X_val, y_val):
        weights = []
        for i, member_clf in enumerate(self.ensemble_):
            y_pred = member_clf.predict(X_val)
            weights.append(accuracy_score(y_val, y_pred))
        self.classificators_weights = weights.copy()

    def predict(self, X, y):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        if self.voting == "vector":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]
        else: #majority and weighted
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))
            pred_ = np.array(pred_)

            weightArray = []
            for i in set(y):
                weightArray.append([i, 0])

            prediction = []
            for j in range(len(pred_[0])):
                for i in range(len(pred_)):
                    for k in weightArray:
                        if (k[0] == pred_[i][j]):
                            k[1] += self.classificators_weights[i]
                maxvalue = 0
                maxclass = 0
                for k in weightArray:
                    if k[1] > maxvalue:
                        maxclass = k[0]
                        maxvalue = k[1]
                    k[1] = 0
                prediction.append(maxclass)

            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
                probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)


class AdaboostEnsembleModel(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, voting="majority", random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        self.classificators_weights = np.ones((n_estimators,), dtype=int)
        np.random.seed(self.random_state)

    def fitmain(self, X, y):
        #Ustalenie wag klasyfikatorów w przypadku głosownaia ważonego
        if self.voting == "weighted":
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.50, random_state=1234)
            self.fit(X_train, y_train)
            self.predictweight(X_val, y_val)
        self.fit(X, y)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        n = X.shape[0]
        sample_weights = np.zeros(shape=(self.n_estimators, n))
        sample_weights[0] = np.ones(shape=n) / n
        self.ensemble_ = []

        for i in range(self.n_estimators):
            curr_sample_weights = sample_weights[i]
            clf = clone(self.base_estimator)
            clf = clf.fit(X, y, sample_weight=curr_sample_weights)

            stump_pred = clf.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()  # / n
            stump_weight = np.log((1 - err+0.5) / (err+0.5)) / 2

            new_sample_weights = (curr_sample_weights * np.exp(-stump_weight * y * stump_pred))
            new_sample_weights /= new_sample_weights.sum()

            if i+1 < self.n_estimators:
                sample_weights[i+1] = new_sample_weights

            self.ensemble_.append(clf)

        return self

    def predictweight(self, X_val, y_val):
        weights = []
        for i, member_clf in enumerate(self.ensemble_):
            y_pred = member_clf.predict(X_val)
            weights.append(accuracy_score(y_val, y_pred))
        self.classificators_weights = weights.copy()

    def predict(self, X, y):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        if self.voting == "vector":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]
        else: #majority and weighted
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))
            pred_ = np.array(pred_)

            weightArray = []
            for i in set(y):
                weightArray.append([i, 0])

            prediction = []
            for j in range(len(pred_[0])):
                for i in range(len(pred_)):
                    for k in weightArray:
                        if (k[0] == pred_[i][j]):
                            k[1] += self.classificators_weights[i]
                maxvalue = 0
                maxclass = 0
                for k in weightArray:
                    if k[1] > maxvalue:
                        maxclass = k[0]
                        maxvalue = k[1]
                    k[1] = 0
                prediction.append(maxclass)

            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
                probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)
        

class RandomSubspaceEnsembleModel(BaseEnsemble, ClassifierMixin):

    def __init__(self,base_estimator=None, n_estimators=10, n_subspace_features=5, voting="majority", random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_subspace_features = n_subspace_features
        self.voting = voting
        self.random_state = random_state
        self.classificators_weights = np.ones((n_estimators,), dtype=int)
        np.random.seed(self.random_state)

    def fitmain(self, X, y):
        #Ustalenie wag klasyfikatorów w przypadku głosownaia ważonego
        if self.voting == "weighted":
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.50, random_state=1234)
            self.fit(X_train, y_train)
            self.predictweight(X_val, y_val)
        self.fit(X, y)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
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

    def predictweight(self, X_val, y_val):
        weights = []
        for i, member_clf in enumerate(self.ensemble_):
            y_pred = member_clf.predict(X_val[:, self.subspaces[i]])
            weights.append(accuracy_score(y_val, y_pred))
        self.classificators_weights = weights.copy()

    def predict(self, X, y):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        if self.voting == "vector":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]
        else: #majority and weighted
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
            pred_ = np.array(pred_)

            weightArray = []
            for i in set(y):
                weightArray.append([i, 0])

            prediction = []
            for j in range(len(pred_[0])):
                for i in range(len(pred_)):
                    for k in weightArray:
                        if (k[0] == pred_[i][j]):
                            k[1] += self.classificators_weights[i]
                maxvalue = 0
                maxclass = 0
                for k in weightArray:
                    if k[1] > maxvalue:
                        maxclass = k[0]
                        maxvalue = k[1]
                    k[1] = 0
                prediction.append(maxclass)

            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)


#MAIN EXPERIMENT
dataset = 'telescope/magic.csv'
dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
dataset[~np.isnan(dataset).any(axis=1), :]
X = dataset[1:, :-1]
X = np.nan_to_num(X)
y = dataset[1:, -1].astype(int)

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

ensemble_methods = ["Adaboost","RandomSubspace", "Bagging"]
base_estimators = [GaussianNB(), DecisionTreeClassifier(), KNeighborsClassifier()]
number_of_classificators = [10]
combination_methods = ["majority",  "vector", "weighted"]

test_results = pd.DataFrame()

wilcoxonAdaboost = np.zeros(((len(base_estimators)-1)*len(combination_methods), n_splits * n_repeats))
wilcoxonRandomSubspace = np.zeros((len(base_estimators)*len(combination_methods), n_splits * n_repeats))
wilcoxonBagging = np.zeros((len(base_estimators)*len(combination_methods), n_splits * n_repeats))

for ensemble_method in range(len(ensemble_methods)):
    for base in range(len(base_estimators)):
        if ensemble_methods[ensemble_method] == "Adaboost" and isinstance(base_estimators[base], KNeighborsClassifier):
            continue
        for number in number_of_classificators:
            for combination in range(len(combination_methods)):

                clf = 0
                if ensemble_methods[ensemble_method] == "RandomSubspace":
                    clf = RandomSubspaceEnsembleModel(base_estimator=base_estimators[base], n_estimators=number, n_subspace_features=math.floor(X.shape[1]/2), voting=combination_methods[combination], random_state=123)
                elif ensemble_methods[ensemble_method] == "Bagging":
                    clf = BaggingEnsembleModel(base_estimator=base_estimators[base], n_estimators=number, voting=combination_methods[combination], random_state=123)
                elif ensemble_methods[ensemble_method] == "Adaboost":
                    clf = AdaboostEnsembleModel(
                        base_estimator=base_estimators[base], n_estimators=number, voting=combination_methods[combination], random_state=123)

                scores = []

                for fold_id, (train, test) in enumerate(rskf.split(X, y)):

                    clf.fitmain(X[train], y[train])
                    y_pred = clf.predict(X[test], y[test])

                    if ensemble_methods[ensemble_method] == "RandomSubspace":
                            wilcoxonRandomSubspace[len(base_estimators)*base+combination, fold_id] = accuracy_score(y[test], y_pred)
                    if ensemble_methods[ensemble_method] == "Bagging":
                            wilcoxonBagging[len(base_estimators)*base+combination, fold_id] = accuracy_score(y[test], y_pred)
                    if ensemble_methods[ensemble_method] == "Adaboost":
                            wilcoxonAdaboost[len(base_estimators)*base+combination, fold_id] = accuracy_score(y[test], y_pred)

                    scores.append([accuracy_score(y[test], y_pred), 
                                recall_score(y[test], y_pred, average='weighted'),
                                precision_score(y[test], y_pred, average='weighted', labels=np.unique(y_pred))])
                print(f"{str(ensemble_methods[ensemble_method])}, {str(base_estimators[base])}, {str(number)}, {str(combination_methods[combination])}"+" = Accuracy score: %.3f (%.3f)" % (
                    np.mean(scores, axis=0)[0], np.std(scores, axis=0)[0]))

                test_results = test_results.append({
                    'Ensemble method': str(ensemble_methods[ensemble_method]),
                    'Model': str(base_estimators[base]),
                    'Number of classificators': str(number),
                    'Voting': str(combination_methods[combination]),
                    'accuracy': round(np.mean(scores, axis=0)[0], 4),
                    'accuracy std': "("+str(round(np.std(scores, axis=0)[0], 4))+")",
                    'recall': round(np.mean(scores, axis=0)[1], 4),
                    'recall std': "("+str(round(np.std(scores, axis=0)[1], 4))+")",
                    'precision': round(np.mean(scores, axis=0)[2], 4),
                    'precision std': "("+str(round(np.std(scores, axis=0)[2], 4))+")"
                }, ignore_index=True)

mean_scores_RandomSubspace = np.mean(wilcoxonRandomSubspace, axis=1).T
print("\nMean scores:\n", mean_scores_RandomSubspace)
print("\nMean scores shape:\n", mean_scores_RandomSubspace.shape)

mean_scores_Bagging = np.mean(wilcoxonBagging, axis=1).T
print("\nMean scores:\n", mean_scores_Bagging)
print("\nMean scores shape:\n", mean_scores_Bagging.shape)

mean_scores_Adaboost = np.mean(wilcoxonAdaboost, axis=1).T
print("\nMean scores:\n", mean_scores_Adaboost)
print("\nMean scores shape:\n", mean_scores_Adaboost.shape)

#Print and save to csv files
test_results = test_results.sort_values(by=['accuracy'])
test_results_csv = test_results[['Ensemble method', 'Model', 'Number of classificators', 'Voting',
                                 'accuracy', 'accuracy std', 'recall', 'recall std', 'precision', 'precision std']]
print(test_results_csv)
test_results_csv.to_csv(r'.\test_results.csv', index = False, header=True)
