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
[0.81192924 ,0.80728477, 0.79346026, 0.75908853 ,0.76107529, 0.75977257, 0.70101516 ,0.70829993, 0.68123475],
[0.07252945, 0.05751457 ,0.23252446,0.98319442, 0.98319442, 0.99081961,0.9677372,  0.9677372,  0.96785451],
[0.86686687 ,0.87567568, 0.87287287 ,0.86806807, 0.86806807, 0.86376376, 0.85705706, 0.85815816, 0.85735736],
[0.48092222 ,0.48539983 ,0.47494193, 0.4958616 , 0.50876018, 0.48472546,0.48092332, 0.49843944, 0.48757466],
[0.91350535, 0.94733746, 0.91703336, 0.91485248, 0.91485248, 0.9333531, 0.82818216, 0.82915455, 0.82760002],
[0.5016, 0.4932, 0.5104 ,0.5052, 0.5012, 0.4984 ,0.4768, 0.494 , 0.514],
[0.86147712, 0.86147712 ,0.86147712 ,0.86046456, 0.84915143, 0.86046538,0.86147712 ,0.86147712 ,0.85885086],
[0.47014925,0.47014925, 0.47412935, 0.51890547, 0.51915423, 0.52064677, 0.48208955, 0.49975124, 0.50422886],
[0.77899335, 0.77804741, 0.77667757 ,0.78845244, 0.80690387 ,0.80087833 , 0.78751892 ,0.8056817 , 0.79441658],
[0.6129, 0.771,  0.6584, 0.633 , 0.6332, 0.6086 ,0.648 , 0.7791, 0.6868],
[0.48202856, 0.48202856, 0.48202856 ,0.51797144 ,0.51797144, 0.51797144, 0.51797144, 0.51797144, 0.51797144],
[0.33378086, 0.33333333, 0.33552469, 0.33864198, 0.34364198, 0.33859568,0.3316821 , 0.33140432, 0.33141975],
[0.77175863, 0.76820835 ,0.77253873, 0.82146111, 0.82168114, 0.81785099, 0.81716095 ,0.8253013 , 0.82205114],
[0.91486068, 0.91052632, 0.90743034, 0.92972136, 0.92732198, 0.92623839,0.93397833, 0.93413313 ,0.93320433],
[0.94485323, 0.93988258 ,0.94614481, 0.95064579, 0.94923679, 0.9504501, 0.95127202, 0.95127202, 0.9511546 ],
[0.72594111, 0.72611987, 0.7277918 , 0.81119874, 0.81119874, 0.81677182, 0.77855941, 0.7955836 , 0.7835857 ],
[0.34759916 ,0.34759916 ,0.34759916, 0.65240084, 0.65240084, 0.65240084, 0.65240084 ,0.65240084 ,0.65240084],
[0.45063025, 0.44989496 ,0.45068277 ,0.87447479, 0.87447479, 0.91271008, 0.61617647, 0.61554622 ,0.60945378],
[0.51699029 ,0.51849515, 0.51946602 ,0.90854369 ,0.90849515, 0.91684466, 0.73834951, 0.74150485 ,0.73800971],
[0.37204007, 0.37673042, 0.37613843, 0.90979053 ,0.90979053, 0.91867031,0.70765027, 0.71179417, 0.71161202],
[0.34759916 ,0.34759916, 0.34759916, 0.65240084 ,0.65240084 ,0.65240084 ,0.65240084, 0.65240084, 0.65240084]







]

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
mean_scores_Bagging = [
[0.8079688,  0.81324503, 0.81126699, 0.77628529, 0.77628529, 0.78358313,0.63175322, 0.63109533, 0.62647264],
[0.09637747 ,0.09634813 ,0.09150874 ,1.0    ,     1.0   ,     1.0,0.96770787, 0.9677372 , 0.96770787],
[0.86686687, 0.87567568,0.87287287, 0.86806807, 0.86806807, 0.86376376, 0.8570570, 0.85815816, 0.85735736],
[0.45241505 ,0.43843155, 0.45743466, 0.52423401 ,0.52423419, 0.5232879 ,0.52110956, 0.53020784 ,0.52124653],
[0.98911337, 0.99027955 ,0.9895021,  1.0 ,        1.0 ,        1.0, 0.82857127, 0.82857202, 0.8277957 ],
[0.4888 ,0.4976 ,0.4936, 0.5228 ,0.5228, 0.514 , 0.5132 ,0.5024 ,0.5096],
[0.86147712 ,0.86147712, 0.86147712 ,0.85055617, 0.85055617, 0.84044412, 0.86147712, 0.86147712 ,0.86107226],
[0.5079602 , 0.50174129 ,0.51044776 ,0.50597015 ,0.51243781, 0.50522388, 0.51666667 ,0.51865672, 0.52139303],
[0.77899335 ,0.77804741 ,0.77667757, 0.78845244, 0.80690387, 0.80087833 ,0.78751892, 0.8056817,  0.79441658],
[0.7904, 0.7941 ,0.79,   0.835 , 0.835,  0.8365 ,0.9113, 0.9135 ,0.9147],
[0.48202856, 0.48202856 ,0.48202856 ,0.51797144 ,0.51797144, 0.51797144, 0.51797144 ,0.51797144, 0.51797144],
[0.33493827 ,0.34018519 ,0.33493827,0.35084877, 0.35026235, 0.3508642,0.33186728, 0.33594136, 0.33044753],
[0.72385613, 0.71806592 ,0.71889593, 0.82754136 ,0.82748136 ,0.82540129, 0.79118952, 0.79368963, 0.79316959],
[0.8997678,  0.899613 ,  0.89481424, 0.93134675, 0.93134675, 0.92600619, 0.93421053 ,0.93421053, 0.93421053],
[0.92818004, 0.92716243 ,0.92677104, 0.94880626, 0.94880626 ,0.94692759, 0.95127202, 0.95127202 ,0.95127202],
[0.72633018 ,0.72578339,0.72628812, 0.86028391 ,0.86028391, 0.85914826, 0.79988433 ,0.80198738, 0.80132492],
[0.34759916 ,0.34759916, 0.34759916, 0.65240084 ,0.65240084 ,0.65240084, 0.65240084, 0.65240084, 0.65240084],
[0.45992647, 0.44511555,0.44490546 ,0.99921218,0.99921218, 0.99921218, 0.62064076, 0.61848739 ,0.61528361],
[0.37917476 ,0.38800971, 0.40820388 ,0.99927184, 0.99927184 ,0.9992233, 0.73417476, 0.73796117, 0.73645631],
[0.37204007, 0.37673042 ,0.37613843 ,0.90979053, 0.90979053 ,0.91867031, 0.70765027, 0.71179417, 0.71161202],
[0.34759916 ,0.34759916 ,0.34759916 ,0.65240084, 0.65240084 ,0.65240084, 0.65240084 ,0.65240084, 0.65240084],




]

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
mean_scores_Adaboost = [
[0.81854305, 0.81786337, 0.82249041, 0.72746166 ,0.72746166, 0.7327466 ],
[0.07449717 ,0.07440918, 0.07446784 ,1.0   , 1.0    ,1.0   ],
[0.87527528, 0.8971972 , 0.87737738, 0.97657658, 0.97657658, 0.97497497],
[0.4647648 , 0.4639505,  0.46408619,0.48363002 ,0.48403745, 0.48431011],
[0.99728004,0.99728004 ,0.99728004 ,1.0   ,  1.0   , 1.0],
[0.4836, 0.4804, 0.4812, 0.4984 ,0.4984 ,0.5008],
[0.86147712, 0.86147712 ,0.86147712, 0.78139737, 0.78139737, 0.77795649],
    [0.86147712 ,0.86147712 ,0.86147712 ,0.78139737 ,0.78139737, 0.77795649],
[0.79900489 ,0.80086606, 0.79880225, 0.78282607, 0.78270936 ,0.77655477],
[0.7904, 0.7941 ,0.79 ,  0.835,  0.835,  0.8365, 0.9113, 0.9135 ,0.9147],
[0.48202856 ,0.48202856 ,0.48202856, 0.51797144 ,0.51797144, 0.51797144],
[0.3079321 , 0.3079321  ,0.3079321  ,0.34564815, 0.34564815 ,0.34564815],
[0.80125986, 0.80106992,0.80007977, 0.79557981, 0.79560981, 0.79220967],
[0.93235294, 0.92979876, 0.93235294, 0.88552632 ,0.88552632, 0.88204334],
[0.94324853 ,0.93984344, 0.94324853, 0.91694716, 0.91694716, 0.91455969],
[0.74967403, 0.74885384, 0.75109359, 0.84484753 ,0.84484753 ,0.84099895],
[0.34759916 ,0.34759916 ,0.34759916, 0.65240084,0.65240084, 0.65240084],
[0.45252101 ,0.45252101, 0.45252101 ,0.99947479 ,0.99947479, 0.99947479],
[0.51873786, 0.51902913 ,0.51917476 ,0.99946602 ,0.99946602, 0.99946602],
[0.38237705, 0.38274135, 0.38315118 ,0.99968124, 0.99968124 ,0.99968124],
[0.34759916 ,0.34759916 ,0.34759916 ,0.65240084 ,0.65240084 ,0.65240084]


 ]

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