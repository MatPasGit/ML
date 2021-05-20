import pandas as pd
import numpy as np
def dataset_stroke():
    #dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
    df =pd.read_csv("datasets/stroke/healthcare-dataset-stroke-data.csv")
    print(df.head())
    #print(df.isna().sum())
    print(df.nunique())
    #print(df.work_type.value_counts())
    #print(df.corr())
    df['gender'].replace( ['Male', 'Female'] ,[0, 1], inplace=True)
    df['ever_married'].replace(["true", "false"], [1,0], inplace=True)
    df.loc[df['work_type'] != ("Private" or  "Self-employed") ] = 3
    df['ever_married'].replace(["Private", "Self-employed"], [0, 1], inplace=True)
    df['Residence_type'].replace(["Urban", "Rural"], [0, 1], inplace=True)
    df['smoking_status'].replace(["smokes", 'formerly smoked',"never smoked", "Unknown"], [0, 1, 2 ,3], inplace=True)
    print("dupa")
    del df['id']
    print(df.head())

    return df

def australia_weather():
    df = pd.read_csv("datasets/rain in austraila/weatherAUS.csv")
    df['RainToday'].replace(["No", "Yes"], [0, 1], inplace=True)
    df['RainTomorrow'].replace(["No", "Yes"], [0, 1], inplace=True)
    print(df.nunique())
    print(df.head())


australia_weather()