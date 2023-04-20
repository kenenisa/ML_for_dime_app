import pandas as pd
import numpy as np
import warnings
import json
import Helpers

from collections import defaultdict
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import date, timedelta, datetime

def main(data):

    # df=pd.read_csv('MaunaLoaDailyTemps.csv',index_col='DATE',parse_dates=True)
    df = data
    df=df.dropna()
    # print('Shape of data',df.shape)
    df.head()


    def adf_test(dataset):
        dftest = adfuller(dataset, autolag = 'AIC')
        # print("1. ADF : ",dftest[0])
        # print("2. P-Value : ", dftest[1])
        # print("3. Num Of Lags : ", dftest[2])
        # print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
        # print("5. Critical Values :")
        # for key, val in dftest[4].items():
        #     print("\t",key, ": ", val)
    adf_test(df['average'])

    #Figure Out Order for ARIMA Model

    warnings.filterwarnings("ignore")

    stepwise_fit = auto_arima(df['average'], 
                          suppress_warnings=True)  
    best_model = tuple(map(int, str(stepwise_fit)[7 : 12].split(',')))       

    stepwise_fit.summary()

    train=df.iloc[:-30]
    test=df.iloc[-30:]

    ## Train the Model
    
    model=ARIMA(train['average'],order=best_model)
    model=model.fit()
    model.summary()

    start=len(train)
    end=len(train)+len(test)-1
    #if the predicted values dont have date values as index, you will have to uncomment the following two commented lines to plot a graph
    #index_future_dates=pd.date_range(start='2018-12-01',end='2018-12-30')
    pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')

    test['average'].mean()


    rmse=sqrt(mean_squared_error(pred,test['average']))
    # print(rmse)

    model2=ARIMA(df['average'],order=best_model)
    model2=model2.fit()

    MAX_PREDICTION_DATE = 7
    current_date = date.today()
    last_date = current_date + timedelta(days = MAX_PREDICTION_DATE)

    #For Future Dates
    index_future_dates=pd.date_range(start=str(current_date),end=str(last_date))
    #print(index_future_dates)
    pred=model2.predict(start=len(df),end=len(df) + MAX_PREDICTION_DATE ,typ='levels').rename('ARIMA Predictions')
    #print(comp_pred)
    pred.index=index_future_dates

    predictions, day = [], 0
    for i in pred:
        predictions.append([current_date + timedelta(days = day), i])
        day += 1

    return predictions

if __name__ == "__main__":
    preditions = []
    json_file = Helpers.get_request("prediction")
    deposit_data, expense_data = defaultdict(list), defaultdict(list)

    for file in json_file:
        address = file['address']
        

        for day in file['days']:
            deposit, expense, date = day['deposit'], day['expense'], day['date']
            deposit_data[address].append([date, deposit['total'], deposit['average'], deposit['max']])
            expense_data[address].append([date, expense['total'], expense['average'], expense['max']])

    deposit_prediction, expense_prediction = defaultdict(list), defaultdict(list)
    for key in deposit_data.keys():
        deposit_prediction[key] = main(pd.DataFrame(deposit_data[key], columns = ['DATE', 'Total', 'average', 'max']))
        expense_prediction[key] = main(pd.DataFrame(expense_data[key], columns = ['DATE', 'Total', 'average', 'max']))
    
    for deposit_key in deposit_data:
        merged_predictions = []
        for key in deposit_prediction.keys():
            merged_predictions.append({"future_date" : deposit_prediction[key][0],
                                        "expense" : expense_prediction[key][1],
                                        "deposit" : expense_prediction[key][1]})

            preditions.append({"address" : address, "predictions" : merged_predictions})
    Helpers.post_prediction(json.dumps(preditions))