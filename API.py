from flask import Flask,request,jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler


now = datetime.now()

Time = [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]

df=pd.read_csv('Processed_crime.csv')

train=df.iloc[:,:-1]
scaler=StandardScaler()

scaler.fit_transform(train)

with open ("model.pkl","rb") as f:
    xgb_clf=pickle.load(f)



app=Flask(__name__)

@app.route('/classify_route',methods=['POST'])
def predict():

    data=request.get_json()

    for route in data:
        location_list=[]
        for i in range(len(route)):
            l = []
            for key in route[i]:
                if key == 'lat' or key == 'lng':
                    l.append(float(route[i][key]))

            location_list.append(l)

        new_df = pd.DataFrame(location_list)
        new_df['Date'] = Time[0]
        new_df['Time'] = Time[1]
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_df['Time'] = pd.to_datetime(new_df['Time'])

        new_df['YEAR'] = new_df['Date'].dt.year
        new_df[2] = new_df['Date'].dt.month
        new_df['DAY'] = new_df['Date'].dt.day

        new_df[3] = new_df['Time'].apply(lambda x: x.hour)
        new_df[4] = new_df['Time'].apply(lambda x: x.minute)
        new_df.drop(columns=['Date', 'Time', 'YEAR', 'DAY'], inplace=True)

        print(new_df)

        new_df_values = scaler.transform(new_df)

        y_pred = xgb_clf.predict(new_df_values)

        print(y_pred)

        counts = np.bincount(y_pred, minlength=10)


        print(counts)

        return counts


app.run()
