from scipy.stats import ranksums
from tabulate import tabulate
import numpy as np
from scipy.stats import rankdata

#Random Subspace ranks
print("Random Subspace ranks")
mean_scores_RandomSubspace = [[0.85735736, 0.85805806, 0.85965966, 0.83153153, 0.83153153, 0.81851852,
                               0.85375375, 0.85925926, 0.85445445],
                              [0.4735736,  0.85805806, 0.85965966, 0.83153153, 0.83153153, 0.81851852,
                               0.85375375, 0.85925926, 0.85445445]]

ranks_RandomSubspace = []
for ms in mean_scores_RandomSubspace:
    ranks_RandomSubspace.append(rankdata(ms).tolist())
ranks_RandomSubspace = np.array(ranks_RandomSubspace)
print("\nRanks:\n", ranks_RandomSubspace)

mean_ranks_RandomSubspace = np.mean(ranks_RandomSubspace, axis=0)
print("\nMean ranks:\n", mean_ranks_RandomSubspace)

alfa = .05
w_statistic = np.zeros((9, 9))
p_value = np.zeros((9,9))

for i in range(9):
    for j in range(9):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_RandomSubspace.T[i], ranks_RandomSubspace.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight",
            "Tree-major","Tree-vect","Tree-weight",
            "KNeigh-major","KNeigh-vect","KNeigh-weight",]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((9, 9))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((9, 9))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)



# Bagging ranks
print("Bagging ranks")
mean_scores_Bagging = []

ranks_Bagging = []
for ms in mean_scores_Bagging:
    ranks_Bagging.append(rankdata(ms).tolist())
ranks_Bagging = np.array(ranks_Bagging)
print("\nRanks:\n", ranks_Bagging)

mean_ranks_Bagging = np.mean(ranks_Bagging, axis=0)
print("\nMean ranks:\n", mean_ranks_Bagging)


alfa = .05
w_statistic = np.zeros((9, 9))
p_value = np.zeros((9,9))

for i in range(9):
    for j in range(9):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_Bagging.T[i], ranks_Bagging.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight",
            "Tree-major","Tree-vect","Tree-weight",
            "KNeigh-major","KNeigh-vect","KNeigh-weight",]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((9, 9))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((9, 9))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)


# Adaboost ranks
print("Adaboost ranks")
mean_scores_Adaboost = []

ranks_Adaboost = []
for ms in mean_scores_Adaboost:
    ranks_Adaboost.append(rankdata(ms).tolist())
ranks_Adaboost = np.array(ranks_Adaboost)
print("\nRanks:\n", ranks_Adaboost)

mean_ranks_Adaboost = np.mean(ranks_Adaboost, axis=0)
print("\nMean ranks:\n", mean_ranks_Adaboost)

alfa = .05
w_statistic = np.zeros((6, 6))
p_value = np.zeros((6,6))

for i in range(6):
    for j in range(6):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_Adaboost.T[i], ranks_Adaboost.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight",
            "Tree-major","Tree-vect","Tree-weight"]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((6, 6))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((6, 6))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)