from scipy import *
import pandas as pd
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np
#Program z ML


#STARY PROGRAM Z I STOPNIA
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import sklearn

# from numpy import set_printoptions
# from sklearn.feature_selection import SelectKBest, f_classif, chi2
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score


# NAMES = ("Koncentracja hemoglobiny", "Liczba erytrocytów", "Średnia objętość krwinki", "Średnie stężenie HB w krwince MCHC (%)", "Wielkość erytrocytów", "Rodzaj erytrocytów",          "Tkanka siateczkowata", "Szpik kostny", "Wielkość", "Stosunek jądrowo-cytoplazmowy", "Rodzaj jądra", "Struktura chromatyny jądrowej", "Jąderko", "Pasożyty",                   "Ziarenka żelaza", "Poziom żelaza", "Poziom trwałych związków żelaza", "Poziom witaminy B12", "Poziom kwasu foliowego", "Nieznany parametr", "Reakcja (test odpornościowy)",
#          "Reakcja (test urobilinowy, urobilinogenowy, urobilirubinowy)", "Reakcja (test ruchliwości komórki)", "Płeć", "Wiek", "Gorączka", "Krwawienie", "Skóra",
#          "Węzły chłonne", "Szmery sercowe", "Wątroba, śledziona", "Diagnoza")
# TARGET = "Diagnoza"
# TARGET_NAMES = ("Niedokrwistość normocytowa", "Niedokrwistość megaloblastyczna", "Niedokrwistość z niedoboru żelaza", "Pierwotna niedokrwistość aplastyczna", "Wtórna niedokrwistość aplastyczna", "Wrodzona sferocytoza", "Wrodzona eliptocytoza", "Wrodzona stomatocytoza", "Akantocytoza", "Niedokrwistość wywołana niedoborem G-6-PD",
#                 "Kinaza pirogronianowa", "Niedokrwistość śródziemnomorska", "Niedokrwistość sierpowatokrwinkowa", "Niedokrwistość autoimmunohemolityczna", "Połowiczna niedokrwitość immunohemolityczna", "Niedokrwistość jatrogenna", "Krwotoczna utrata krwi", "Krwotok wywoałny ankilostomozą", "Krwotok wywołany wrzodem jelita", "Krwotok wywołany nadżerką")
# data = pd.read_csv("Dane.csv", names=NAMES)
# #pd.set_option('display.max_columns', None)
# #pd.set_option('display.max_rows', None)

# print("Wykresy cech\n")
# fig, axs = plt.subplots(8, 4, figsize=(20, 30), facecolor='w', edgecolor='k')
# fig.subplots_adjust(hspace=.6, wspace=.3)
# axs = axs.ravel()

# for i in range(32):
#     axs[i].set_title(NAMES[i])
#     labels, counts = np.unique(data[NAMES[i]].values, return_counts=True)
#     axs[i].bar(labels, counts, align='center')
#     axs[i].set_xticks(labels)
# plt.show()

# array = data.values
# X = array[:, :-1]
# y = array[:, -1].astype(int)
# fit = SelectKBest(f_classif).fit(X, y)
# set_printoptions(precision=3)
# print("Wyniki dla f_classif\n")
# print(fit.scores_)
# #fit = SelectKBest(chi2).fit(X, y)
# #set_printoptions(precision=3)
# #print("Wyniki dla chi2\n")
# #print(fit.scores_)

#DECISION TREE
    read = pd.read_csv("datasets/credit_risk/original.csv")
    print(read.head())
    klasyfikator= DecisionTreeClassifier(criterion="entropy")
    X = read.drop('default', axis = 1)
    y = read['default']
    np.where(y.values >= np.finfo(y.float64).max)
    print(y)
    klasyfikator.fit(X = X, y = y)
# from scipy.stats import ttest_ind
# from tabulate import tabulate

# #Importujemy funkcje potrzebne do tworzenia wykresów
# from bokeh.io import push_notebook, show, output_notebook
# from bokeh.layouts import row, gridplot
# from bokeh.plotting import figure, show, output_file
# from bokeh.models import Span
# output_notebook()

# #Tworzymy listę narzędzi, które chcemy użyć w Naszym wykresach
# TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

# def neuronnetwork(X, y, alfa, neurons, splits, repeats, actualFeatures, numberOfFeatures):

#     global bestScore, AllScores
#     cross = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=87654321)    # 5 powtorzen 2-krotnej walidacja krzyżowa
#     network = MLPClassifier(hidden_layer_sizes=neurons, momentum=alfa, random_state=87654321)     # Inicjalizacja modelu sieci neuronowej
#     scores=[]
#     firstScore = True

#     for train_index, test_index in cross.split(X, y):
#         x_train, x_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         network.fit(x_train, y_train)
#         predict = network.predict(x_test)
#         score = accuracy_score(y_test, predict)
#         confusionMatrix = confusion_matrix(y_test,predict)
#         scores.append(score)            # Wypisanie wyniku
#         if firstScore:
#             confusionMatrixMean = confusionMatrix
#             firstScore = False
#         else:
#             confusionMatrixMean += confusionMatrix

#     allScores.append(scores)
#     mean_score = np.mean(scores)
#     std_score = np.std(scores)
#     confusionMatrixMean = confusionMatrixMean/(repeats)
#     print(f"\nSrednia: {mean_score}\nOdchylenie: {std_score}\n")

#     scoresToPlot.append(mean_score)

#     if bestScore[4] < mean_score:
#         bestScore= [neurons, alfa, numberOfFeatures, actualFeatures, mean_score, std_score, confusionMatrixMean]



# neurons = [200, 250, 300]        # Liczby neuronów w warstwie ukrytej (50, 100, 150)
# momentum = [0, 0.9]             # Wartości współczynnika momentu (0 - momentum nieaktywne; 0,9- najpopularniejsza wartość momentum)
# splits = 2                      # Krotność walidacji (2)
# repeats = 5                     # Powtórzenia w walidacji (5)
# bestScore = [0, 0, 0, 0, 0, 0, np.ndarray ]  # Najlepszy wynik -> liczba neuronów, wartość momentum, liczba cech, cechy, wynik działania sieci, odchylenie, macierz konfuzji

# # Badania przeprowadzane przy uzyciu cech wskazanych podczas rankingu cech wykonanych wcześniej - fclassif
# #features=[10,5,4,2,7,14,25,11,1,16]
# #features=[10,5,4,2,7,14,25,11,1,16,12,8,29,3,13,18,6,15,22,9]
# features=[10,5,4,2,7,14,25,11,1,16,12,8,29,3,13,18,6,15,22,9,26,0,17,23,19,21,20,28,24,30,27]

# maxNumberOfFeatures = len(features)        # Maksymalna liczba cech użyta w badaniu
# allMethods = len(neurons)*len(momentum)*maxNumberOfFeatures
# arrayOfFeatures=range(1,maxNumberOfFeatures+1)
# headers = list(map(str,arrayOfFeatures))
# names_column=[]

# for i in arrayOfFeatures:
#     temporalArray=[]
#     temporalArray.append(str(i))
#     names_column.append(temporalArray)
# names_column = np.array(names_column)

# #Tworzenie wykresu
# p = figure(title="Wyniki eksperymentu dla liczby cech równej "+str(maxNumberOfFeatures), tools=TOOLS, plot_width=1200)
# p.xaxis.axis_label = "Liczba cech"
# p.yaxis.axis_label = "Wynik"

# colors= ['black', 'blue', 'yellowgreen', 'green', 'coral', 'red', 'firebrick', 'fuchsia', 'gold', 'gray', 'khaki', 'navy', 'violet' ]
# colorNumber=0

# for neuron in neurons:
#     for alfa in momentum:

#         allScores = []
#         scoresToPlot=[]

#         for numberOfFeatures in range(1, maxNumberOfFeatures+1):
#             actualFeatures = features[:numberOfFeatures]
#             print("Cechy: "+str(actualFeatures))
#             print("Liczba neuronow:"+str(neuron)+", wartosc momentu:"+str(alfa)+", liczba cech:"+str(numberOfFeatures))
#             neuronnetwork(X[:, actualFeatures], y, alfa, neuron, splits, repeats, actualFeatures, numberOfFeatures)

#         p.line(arrayOfFeatures, scoresToPlot, legend_label="Liczba neuronow:"+str(neuron)+", wartosc momentu:"+str(alfa), color=colors[colorNumber])
#         colorNumber +=1
#         # Statystyka - testy parowe
#         print("Testy parowe dla liczby neuronow:"+str(neuron)+", wartosc momentu:"+str(alfa))

#         alfa = 0.05
#         t_statistic = np.zeros((maxNumberOfFeatures,maxNumberOfFeatures))
#         p_value = np.zeros((maxNumberOfFeatures,maxNumberOfFeatures))

#         for i in range(maxNumberOfFeatures):
#             for j in range(maxNumberOfFeatures):
#                 t_statistic[i, j], p_value[i, j] = ttest_ind(allScores[i], allScores[j])


#         t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
#         t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
#         p_value_table = np.concatenate((names_column, p_value), axis=1)
#         p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

#         advantage = np.zeros((maxNumberOfFeatures,maxNumberOfFeatures))
#         advantage[t_statistic > 0] = 1
#         advantage_table = tabulate(np.concatenate(
#             (names_column, advantage), axis=1), headers)

#         significance = np.zeros((maxNumberOfFeatures,maxNumberOfFeatures))
#         significance[p_value <= alfa] = 1
#         significance_table = tabulate(np.concatenate(
#             (names_column, significance), axis=1), headers)

#         stat_better = significance * advantage
#         stat_better_table = tabulate(np.concatenate(
#             (names_column, stat_better), axis=1), headers)
#         print("Statystycznie znacząco lepsze:\n", stat_better_table)


#print(f"\nNajlepszy wynik:\nLiczba neuronów: {bestScore[0]}\nMomentum: {bestScore[1]}\nLiczba cech: {bestScore[2]}\nCechy: {bestScore[3]}\n Wynik: {bestScore[4]}\nOdchylenie: {bestScore[5]}\nMacierz konfuzji:\n {bestScore[6]}\n")
#show(p)


#NOWY PROGRAM