from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from functools import partial
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

app = Flask(__name__)

# index type data
nasdaq_raw = pd.read_csv("Nasdaq Futures.csv")
dow_raw = pd.read_csv("Dow Futures.csv")
russell_raw = pd.read_csv("Russell 2000 Futures.csv")
sp_raw = pd.read_csv("S_P Futures.csv")

# COVID-19 data
virus_data = pd.read_csv("virusdata_preprocess.csv")
virus_data = virus_data.set_index("Date")

def stockdata_preprocess(data_raw, stock_name, virus=True):
    
    ## preprocessing data
    data_prep = data_raw.dropna()
    data_prep = data_prep.sort_values(by=["Date"])
    data_prep = data_prep.set_index("Date")
    
    data_prep = data_prep.drop(["Open", "High", "Low", "Adj Close"], axis=1)

    # Moving Average
    data_prep["Ave10Price"] = data_prep.iloc[:,0].rolling(window=10).mean()
    data_prep["Ave10Volume"] = data_prep.iloc[:,1].rolling(window=10).mean()
    data_prep["Stdev10Price"] = data_prep.iloc[:,0].rolling(window=10).std()
    data_prep["Stdev10Volume"] = data_prep.iloc[:,1].rolling(window=10).std()

    # Bollinger
    data_prep["Bollinger"] = data_prep["Ave10Price"] + 2*data_prep["Stdev10Volume"]

    # Oscillator
    data_prep["Max5Price"] = data_prep.iloc[:,0].rolling(window=5).max()
    data_prep["Min5Price"] = data_prep.iloc[:,0].rolling(window=5).min()
    data_prep["K"] = (data_prep["Close"] - data_prep["Min5Price"])/(data_prep["Max5Price"] - data_prep["Min5Price"])
    data_prep["Oscillator"] = data_prep.iloc[:,-1].rolling(window=3).mean()
    data_prep = data_prep.drop(["Max5Price", "Min5Price", "K"], axis=1)
    
    # Change to rate of change
    data_roc = data_prep.pct_change()
    data_roc = data_roc.drop(["Oscillator"], axis=1)
    data_roc.columns = ["CPrice", "CVolume","CAve10Price","CAve10Volume",
                        "CStdev10Price","CStdev10Volume","CBollinger"]
    
    # cleaning delete Nan, inf -> 0, volume >(<) (-)10 -> (-)10
    data_roc_cleaned = data_roc.copy().dropna()
    data_roc_cleaned = data_roc_cleaned.replace(np.inf, 1)
    data_roc_cleaned = data_roc_cleaned.replace(-np.inf, 1)
    data_roc_cleaned["CVolume"] = np.where(data_roc_cleaned.CVolume > 10, 10, data_roc_cleaned.CVolume)
    data_roc_cleaned["CVolume"] = np.where(data_roc_cleaned.CVolume < -10, -10, data_roc_cleaned.CVolume)
    
    # Target Column
    data_roc_cleaned["Target"] = np.where(data_roc_cleaned.CPrice > 0, 1, 0)
    data_roc_cleaned["Target"] = data_roc_cleaned["Target"].shift(periods=-1) # have to shift for prediction
    
    # merge feature data
    if virus == True:
        data_input = pd.concat([data_roc_cleaned, data_prep, virus_data], axis=1, join='inner')
    else:
        data_input = pd.concat([data_roc_cleaned, data_prep], axis=1, join='inner')
    
    data_input = data_input.dropna()
    data_input["Name"] = stock_name
    
    # delete unused column
    data_input = data_input.drop(["Close", "Volume","Ave10Price","Ave10Volume",
                                 "Stdev10Price","Stdev10Volume","Bollinger"], axis=1)

    return data_input


# ### Preparing index type data

dow_input = stockdata_preprocess(dow_raw, "Dow")
nasdaq_input = stockdata_preprocess(nasdaq_raw, "Nasdaq")
russell_input = stockdata_preprocess(russell_raw, "Russell")
sp_input = stockdata_preprocess(sp_raw, "S&P")

# concat all data
index_input = pd.concat([dow_input, sp_input, nasdaq_input, russell_input], axis=0)


# ## Building logistic regression model

# Spliting 
index_input_y = index_input["Target"]
test_date = '2020-04-01'

# training data
X_train_base = index_input.loc[index_input.index < test_date]
X_train = X_train_base.drop(["Name","Target"], axis=1)
y_train = index_input_y.loc[index_input_y.index < test_date]

# values -> standardized
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# Logistic regression
reg_index = LogisticRegression(penalty="l1", solver="liblinear", class_weight = "balanced", random_state=0)
reg_index.fit(X_train_std, y_train)

# standization pickle
# sc_pickle = pickle.dumps(sc)
sc_file = 'sc.sav'
pickle.dump(sc, open(sc_file, 'wb'))

# model pickle 
#model_pickle = pickle.dumps(reg_index)
model_file = 'model.sav'
pickle.dump(reg_index, open(model_file, 'wb'))


# ## For prediction (not for building model)
# Reference to call model prediction

# loading pickled model
reg_loaded = pickle.load(open(model_file, 'rb'))
sc_loaded = pickle.load(open(sc_file, 'rb'))

'''
#create api
@app.route('/api/', methods=['GET', 'POST'])
@app.route('/api',methods=['GET', 'POST'])
def predict():

    data = request.get_json(force=True)
    requestData = [data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]
    requestData = np.array([requestData])

    d = {'CPrice': [-0.026509], 'CVolume': [0.245612], 'CAve10Price': [0.005099], 'CAve10Volume': [0.112444],
    'CStdev10Price': [-0.022922], 'CStdev10Volume': [0.120680], 'CBollinger': [0.120655], 'Oscillator': [0.337771],
    'Cases_roc': [0.133920], 'Deaths_roc': [0.228247]}

    X_pred  = sc_loaded.transform(pd.DataFrame(d))

    y_pred = reg_loaded.predict(X_pred) # Hard prediction
    pred_probs = reg_loaded.predict_proba(X_pred) # Soft prediction

    # 1 = UP, 0 = Down, probs = percentage of UP
    # print(y_pred[0], pred_probs[:,1][0])

    return Response(json.dumps(int(y_pred[0])))'''
    
#create api
@app.route('/api/', methods=['GET', 'POST'])
@app.route('/api',methods=['GET', 'POST'])
def predict():
    # Get the data from POST request
    data = request.get_json(force=True)
    requestData = [data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]
    requestData = np.array([requestData])

    # Make prediction using model 
    prediction = rfc_model.predict(requestData)
    return Response(json.dumps(int(prediction[0])))

if __name__ == '__main__':
   app.run()