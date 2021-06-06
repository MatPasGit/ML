import re

import pandas as pd
import stopwords as stopwords


def mobilepriceClassification():

    #dane testowe nie mają labelek z przedziałem ceny
    X = pd.read_csv('datasets/MobilePriceClassification/train.csv')
    #df2 = pd.read_csv('datasets/MobilePriceClassification/test.csv')

    #df2.drop(['id'],axis='columns', inplace=True)
    #print(df2.head())

    #df=pd.concat([df1,df2])
    X.dropna()
    y=X['price_range']
    X.drop(['price_range'], axis='columns', inplace=True)

    #print(df.head())
    return X,y

def income():
    X = pd.read_csv('datasets/income/income_evaluation.csv')
    X.dropna()
    X[" income"].replace({" <=50K":"0", " >50K":"1"}, inplace=True)#spacje są w pliku, to ze spacjami trzeba zmieniać :/
    y=X[" income"]
    X.drop([' income'], axis='columns', inplace=True)
    #print(y.head)
    return X,y

def mushrooms():
    X = pd.read_csv('datasets/mushrooms/mushrooms.csv')
    X.dropna()
    y= X["class"]
    X.drop(['class'], axis='columns', inplace=True)
    return X,y

def nursery():
    X = pd.read_csv('datasets/nursery/nursery.csv')
    X.dropna()
    #print(X)
    y = X["Class"]
    X.drop(["Class"], axis='columns', inplace=True)
    #print(X.head())
    #print(y.head())
    return X,y

def rainInAUS():
    X = pd.read_csv('datasets/rain in austraila/weatherAUS.csv')
    X.dropna()
    y = X["RainTomorrow"]
    X.drop(["RainTomorrow"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X,y

def stroke():
    X = pd.read_csv('datasets/stroke/healthcare-dataset-stroke-data.csv')
    X.dropna()
    y = X["stroke"]
    X.drop(["stroke"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X,y

def telescope(): #ready
    X = pd.read_csv('datasets/telescope/magic.csv')
    X.dropna(how='all')
    X["Class"].replace({"g": "0", "h": "1"}, inplace=True)
    y = X["Class"]
    X.drop(["Class"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X,y

def titanic():
    X = pd.read_csv('datasets/titanic/full.csv')
    X.dropna(thresh=2)
    #me,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,WikiId,Name_wiki,Age_wiki,Hometown,Boarded,Destination,Lifeboat,Body,Class
    X.drop(["Name"], axis='columns', inplace=True)#Niepotrzebne
    X["Sex"] = X["Sex"].str.replace(r'\W', "")#usuwa wszystko co nie jest literą ani numerem
    X["Hometown"] = X["Hometown"].str.replace(r'\W', "")
    X["Boarded"] = X["Boarded"].str.replace(r'\W', "")
    X["Destination"] = X["Destination"].str.replace(r'\W', "")
    X["Lifeboat"] = X["Lifeboat"].str.replace(r'\W', "")
    y = X["Survived"]
    X.drop(["Survived"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X,y

def water_potability():
    X = pd.read_csv('datasets/waterqual/water_potability.csv')
    X.dropna(how='all')
    y = X["Potability"]
    X.drop(["Potability"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X,y

def wine_qual():
    X = pd.read_csv('datasets/wine/winequality-red.csv')
    X.dropna(how='all')
    y = X["quality"]
    X.drop(["quality"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def heart():
    X = pd.read_csv('datasets/1/heart.csv')
    X.dropna(how='all')
    y = X["target"]
    X.drop(["target"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def bancrupcy():
    X = pd.read_csv('datasets/bancrupcy/data.csv')
    X.dropna(how='all')
    y = X["Bankrupt?"]
    X.drop(["Bankrupt?"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def contraceptive():
    X = pd.read_csv('datasets/contraceptive/contraceptive.csv')
    X.dropna(how='all')
    y = X["Contraceptive_method"]
    X.drop(["Contraceptive_method"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def employee_attrition(): #14
    X = pd.read_csv('datasets/employee-attrition/employee_attrition_train.csv')
    X.dropna(how='all')
    X["Attrition"].replace({"No": "0", "Yes": "1"}, inplace=True)
    y = X["Attrition"]
    X.drop(["Attrition"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def employee_satisfaction(): #15
    X = pd.read_csv('datasets/employee-satisfaction/Employee Satisfaction Index.csv')
    X.drop(["record"], axis='columns', inplace=True)
    X.dropna(how='all')
    y = X["satisfied"]
    X.drop(["satisfied"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y


def server_logs(): #16
    X = pd.read_csv('datasets/serverlogs/CIDDS-001-external-week1.csv')
    X[" income"].replace({"suspicious": "1", "normal": "0"},inplace=True)
    X.drop(["record"], axis='columns', inplace=True)
    X.dropna(how='all')
    y = X["class"]
    X.drop(["class"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y


def fake_news(): #17
    X = pd.read_csv('datasets/fakenews/FA-KES-Dataset.csv')
    y = X["labels"]
    X.drop(["labels"], axis='columns', inplace=True)
    #print(X.head())
    X["article_content"] = X["article_content"].str.replace(r'\W', " ")
    X["article_content"] = X["article_content"].str.replace(r'\n', "")
    X["article_content"] = X["article_content"].str.replace(r' +', " ")
    X["article_content"] = X["article_content"].str.replace(r'^ ', "")
    X["article_content"] = X["article_content"].str.replace(r' $', "")
    X["article_content"] = X["article_content"].str.replace(r',', "")
    X["article_content"] = X["article_content"].str.replace(r'-', "")

    #print(X.head())
    #print(y.head())
    return X, y

def server_logs(): #18
    X = pd.read_csv('datasets/serverlogs/CIDDS-001-external-week1.csv')
    X[" income"].replace({"suspicious": "1", "normal": "0"},inplace=True)
    X.drop(["record"], axis='columns', inplace=True)
    X.dropna(how='all')
    y = X["class"]
    X.drop(["class"], axis='columns', inplace=True)
    print(X.head())
    print(y.head())
    return X, y

def disease(): #19
    data = pd.read_csv('datasets/disease/dataset.csv')
    severity = pd.read_csv('datasets/disease/Symptom-severity.csv')
    cols = [i for i in data.iloc[:, 1:].columns]
    tmp = pd.melt(data.reset_index(), id_vars=['index'], value_vars=cols)
    tmp['add1'] = 1
    tmp.head()
    X = pd.pivot_table(tmp, values='add1',index='index',columns='value')
    X.insert(0, 'label', data['Disease'])
    X = X.fillna(0)
    y = X["label"]
    X.drop(["label"], axis='columns', inplace=True)
    #print(X.head())
    #print(y.head())
    return X, y



def stocks14():
    stock =pd.read_csv('datasets/usstocks/2014_Financial_Data.csv')
    stock.dropna(how='all')
    y = stock["Class"]
    stock.drop(["Class"], axis='columns', inplace=True)
    print(stock.head())
    print(y.head())
    return stock, y

def stocks15():
    stock =pd.read_csv('datasets/usstocks/2015_Financial_Data.csv')
    stock.dropna(how='all')
    y = stock["Class"]
    stock.drop(["Class"], axis='columns', inplace=True)
    print(stock.head())
    print(y.head())
    return stock, y

def stocks16():
    stock =pd.read_csv('datasets/usstocks/2016_Financial_Data.csv')
    stock.dropna(how='all')
    y = stock["Class"]
    stock.drop(["Class"], axis='columns', inplace=True)
    print(stock.head())
    print(y.head())
    return stock, y

def stocks17():
    stock =pd.read_csv('datasets/usstocks/2017_Financial_Data.csv')
    stock.dropna(how='all')
    y = stock["Class"]
    stock.drop(["Class"], axis='columns', inplace=True)
    print(stock.head())
    print(y.head())
    return stock, y

def stocks18():
    stock =pd.read_csv('datasets/usstocks/2018_Financial_Data.csv')
    stock.dropna(how='all')
    y = stock["Class"]
    stock.drop(["Class"], axis='columns', inplace=True)
    print(stock.head())
    print(y.head())
    return stock, y



