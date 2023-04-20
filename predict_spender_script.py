import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import sklearn
import time
import Helpers
import json

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pandas import json_normalize



def run():

    json_file = Helpers.get_request("reserve")
    if not json_file:
        return
    df = pd.DataFrame.from_dict(data = json_file)
    # df = pd.read_csv('test.csv')

    df.isnull()
    df.isnull().sum().sum()
    df.dropna(inplace=True)
    df['active'].replace({False, True}, inplace=True)

    df['spender'] = [1 if score > 0 else 0 for score in df['credit_score']]

    X = df.drop(['credit_score','spender', 'address'], axis = 1)

    y = df['spender']

    # Normalize feature variablesso that they will have a mean value 0 and standard deviation of 1. 

    X_features = X

    X=StandardScaler().fit_transform(X)

    # Splitting the data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)



    model1 = DecisionTreeClassifier(random_state=1)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    # print(classification_report(y_test, y_pred1))


    model2 = RandomForestClassifier(random_state=1)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    # print(classification_report(y_test, y_pred2))

    # Filtering df for only good quality
    high_spender_count = len(df[df['spender']==1])

    # Filtering df for only good quality
    low_spender_count = len(df[df['spender']==0])

    Helpers.post_reserver(high_spender_count, low_spender_count)
    # print(high_spender_count, low_spender_count)

def run_script():
    run()
