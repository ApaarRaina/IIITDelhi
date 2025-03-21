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
train.drop(columns=['Unnamed: 0'],inplace=True)

scaler=StandardScaler()

scaler.fit_transform(train)

with open ("model.pkl","rb") as f:
    xgb_clf=pickle.load(f)



app=Flask(__name__)


@app.route('/classify_route', methods=['POST'])
def predict():
    data = request.get_json()

    # Initialize an empty list to store results for all routes
    results = []

    # Assuming data is a list of routes
    for route in data:
        location_list = []
        for point in route:
            l = [float(point['lat']), float(point['lng'])]
            location_list.append(l)

        # Create DataFrame with proper column names
        new_df = pd.DataFrame(location_list, columns=[0, 1])
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

        # Make sure columns match what the model was trained on
        new_df = new_df[[0, 1, 2, 3, 4]]  # Ensure column order

        new_df_values = scaler.transform(new_df)
        y_pred = xgb_clf.predict(new_df_values)

        sum_val = np.sum(y_pred)
        div = len(y_pred) * 10
        score = float(sum_val / div)

        results.append(score)

    # Return the first result if there's only one route, or all results
    if len(results) == 1:
        return jsonify(results[0])
    else:
        return jsonify(results)


app.run(debug=True)


