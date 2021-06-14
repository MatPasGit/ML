from scipy.stats import ranksums
from tabulate import tabulate
import numpy as np
from scipy.stats import rankdata


#[0.81192924, 0.80728477,0.79346026 ,0.75908853, 0.76107529, 0.75977257 ,0.70101516, 0.70829993 ,0.68123475],
#[0.8079688 , 0.81324503 ,0.81126699 ,0.77628529, 0.77628529, 0.78358313, 0.63175322, 0.63109533 ,0.62647264],
#[0.81854305, 0.81786337, 0.82249041, 0.72746166, 0.72746166 ,0.7327466 ]
#Random Subspace ranks
print("Random Subspace ranks")
mean_scores_RandomSubspace = [
[ 0.70101516 ,0.70829993, 0.68123475],
[0.9677372,  0.9677372,  0.96785451],
[0.85705706, 0.85815816, 0.85735736],
[0.48092332, 0.49843944, 0.48757466],
[ 0.82818216, 0.82915455, 0.82760002],
[0.4768, 0.494 , 0.514],
[0.86147712 ,0.86147712 ,0.85885086],
[ 0.48208955, 0.49975124, 0.50422886],
[ 0.78751892 ,0.8056817 , 0.79441658],
[0.648 , 0.7791, 0.6868],
[ 0.51797144, 0.51797144, 0.51797144],
[0.3316821 , 0.33140432, 0.33141975],
[ 0.81716095 ,0.8253013 , 0.82205114],
[0.93397833, 0.93413313 ,0.93320433],
[ 0.95127202, 0.95127202, 0.9511546 ],
[ 0.77855941, 0.7955836 , 0.7835857 ],
[ 0.65240084 ,0.65240084 ,0.65240084],
[ 0.61617647, 0.61554622 ,0.60945378],
[ 0.73834951, 0.74150485 ,0.73800971],
[0.70765027, 0.71179417, 0.71161202],
[0.65240084, 0.65240084, 0.65240084]
]

ranks_RandomSubspace = []
for ms in mean_scores_RandomSubspace:
    ranks_RandomSubspace.append(rankdata(ms).tolist())
ranks_RandomSubspace = np.array(ranks_RandomSubspace)
print("\nRanks:\n", ranks_RandomSubspace)

mean_ranks_RandomSubspace = np.mean(ranks_RandomSubspace, axis=0)
print("\nMean ranks:\n", mean_ranks_RandomSubspace)

alfa = .05
w_statistic = np.zeros((3, 3))
p_value = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_RandomSubspace.T[i], ranks_RandomSubspace.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight"]#,
            #"Tree-major","Tree-vect","Tree-weight",
            #"KNeigh-major","KNeigh-vect","KNeigh-weight",]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((3, 3))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((3, 3))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)



# Bagging ranks
print("Bagging ranks")
mean_scores_Bagging =  [
[0.63175322, 0.63109533, 0.62647264],
[0.96770787, 0.9677372 , 0.96770787],
[ 0.8570570, 0.85815816, 0.85735736],
[0.52110956, 0.53020784 ,0.52124653],
[ 0.82857127, 0.82857202, 0.8277957 ],
[ 0.5132 ,0.5024 ,0.5096],
[ 0.86147712, 0.86147712 ,0.86107226],
[ 0.51666667 ,0.51865672, 0.52139303],
[0.78751892, 0.8056817,  0.79441658],
[0.9113, 0.9135 ,0.9147],
[ 0.51797144 ,0.51797144, 0.51797144],
[0.33186728, 0.33594136, 0.33044753],
[ 0.79118952, 0.79368963, 0.79316959],
[ 0.93421053 ,0.93421053, 0.93421053],
[ 0.95127202, 0.95127202 ,0.95127202],
[ 0.79988433 ,0.80198738, 0.80132492],
[ 0.65240084, 0.65240084, 0.65240084],
[ 0.62064076, 0.61848739 ,0.61528361],
[ 0.73417476, 0.73796117, 0.73645631],
[ 0.70765027, 0.71179417, 0.71161202],
[ 0.65240084 ,0.65240084, 0.65240084],
]

ranks_Bagging = []
for ms in mean_scores_Bagging:
    ranks_Bagging.append(rankdata(ms).tolist())
ranks_Bagging = np.array(ranks_Bagging)
print("\nRanks:\n", ranks_Bagging)

mean_ranks_Bagging = np.mean(ranks_Bagging, axis=0)
print("\nMean ranks:\n", mean_ranks_Bagging)


alfa = .05
w_statistic = np.zeros((3, 3))
p_value = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_Bagging.T[i], ranks_Bagging.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight"]#,
            #"Tree-major","Tree-vect","Tree-weight",
            #"KNeigh-major","KNeigh-vect","KNeigh-weight",]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((3, 3))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((3, 3))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)


# Adaboost ranks
print("Adaboost ranks")
mean_scores_Adaboost = [
[ 0.72746166 ,0.72746166, 0.7327466 ],
[1.0   , 1.0    ,1.0   ],
[ 0.97657658, 0.97657658, 0.97497497],
[0.48363002 ,0.48403745, 0.48431011],
[1.0   ,  1.0   , 1.0],
[0.4984 ,0.4984 ,0.5008],
[ 0.78139737, 0.78139737, 0.77795649],
[0.78139737 ,0.78139737, 0.77795649],
[ 0.78282607, 0.78270936 ,0.77655477],
[ 0.5119403,  0.50522388 ,0.50895522],
[ 0.51797144 ,0.51797144, 0.51797144],
[0.34564815, 0.34564815 ,0.34564815],
[ 0.79557981, 0.79560981, 0.79220967],
[ 0.88552632 ,0.88552632, 0.88204334],
[ 0.91694716, 0.91694716, 0.91455969],
[ 0.84484753 ,0.84484753 ,0.84099895],
[ 0.65240084,0.65240084, 0.65240084],
[0.99947479 ,0.99947479, 0.99947479],
[0.99946602 ,0.99946602, 0.99946602],
[0.99968124, 0.99968124 ,0.99968124],
[0.65240084 ,0.65240084 ,0.65240084]

 ]

ranks_Adaboost = []
for ms in mean_scores_Adaboost:
    ranks_Adaboost.append(rankdata(ms).tolist())
ranks_Adaboost = np.array(ranks_Adaboost)
print("\nRanks:\n", ranks_Adaboost)

mean_ranks_Adaboost = np.mean(ranks_Adaboost, axis=0)
print("\nMean ranks:\n", mean_ranks_Adaboost)




alfa = .05
w_statistic = np.zeros((3, 3))
p_value = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks_Adaboost.T[i], ranks_Adaboost.T[j])

headers = ["Gaus-major","Gaus-vect","Gaus-weight"]#,
            #"Tree-major","Tree-vect","Tree-weight",
            #"KNeigh-major","KNeigh-vect","KNeigh-weight",]
names_column = np.expand_dims(np.array(headers), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((3, 3))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((3, 3))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)