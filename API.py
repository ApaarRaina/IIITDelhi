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


#@app.route('/classify_route', methods=['POST'])
def predict():
    #data = request.get_json()

    data=[[
  {
    "idx": 0,
    "lat": "41.91935",
    "lng": "-87.63428"
  },
  {
    "idx": 10,
    "lat": "41.91909",
    "lng": "-87.63460"
  },
  {
    "idx": 20,
    "lat": "41.91769",
    "lng": "-87.63430"
  },
  {
    "idx": 30,
    "lat": "41.91587",
    "lng": "-87.63312"
  },
  {
    "idx": 40,
    "lat": "41.91423",
    "lng": "-87.63191"
  },
  {
    "idx": 50,
    "lat": "41.91311",
    "lng": "-87.63056"
  },
  {
    "idx": 60,
    "lat": "41.91225",
    "lng": "-87.62851"
  },
  {
    "idx": 70,
    "lat": "41.91232",
    "lng": "-87.62688"
  },
  {
    "idx": 80,
    "lat": "41.91219",
    "lng": "-87.62647"
  },
  {
    "idx": 90,
    "lat": "41.91069",
    "lng": "-87.62598"
  },
  {
    "idx": 100,
    "lat": "41.90491",
    "lng": "-87.62486"
  },
  {
    "idx": 110,
    "lat": "41.90218",
    "lng": "-87.62378"
  },
  {
    "idx": 120,
    "lat": "41.90128",
    "lng": "-87.62167"
  },
  {
    "idx": 130,
    "lat": "41.90063",
    "lng": "-87.61942"
  },
  {
    "idx": 140,
    "lat": "41.89396",
    "lng": "-87.61491"
  },
  {
    "idx": 150,
    "lat": "41.89227",
    "lng": "-87.61420"
  },
  {
    "idx": 160,
    "lat": "41.88890",
    "lng": "-87.61420"
  },
  {
    "idx": 170,
    "lat": "41.88536",
    "lng": "-87.61408"
  },
  {
    "idx": 180,
    "lat": "41.88404",
    "lng": "-87.61465"
  },
  {
    "idx": 190,
    "lat": "41.88256",
    "lng": "-87.61703"
  },
  {
    "idx": 200,
    "lat": "41.88108",
    "lng": "-87.61751"
  },
  {
    "idx": 210,
    "lat": "41.87592",
    "lng": "-87.61739"
  },
  {
    "idx": 220,
    "lat": "41.87016",
    "lng": "-87.61724"
  },
  {
    "idx": 230,
    "lat": "41.86863",
    "lng": "-87.61794"
  },
  {
    "idx": 240,
    "lat": "41.86726",
    "lng": "-87.62043"
  },
  {
    "idx": 250,
    "lat": "41.86499",
    "lng": "-87.61956"
  },
  {
    "idx": 260,
    "lat": "41.86500",
    "lng": "-87.61761"
  },
  {
    "idx": 270,
    "lat": "41.86504",
    "lng": "-87.61409"
  },
  {
    "idx": 280,
    "lat": "41.86602",
    "lng": "-87.61381"
  },
  {
    "idx": 290,
    "lat": "41.86610",
    "lng": "-87.61013"
  },
  {
    "idx": 300,
    "lat": "41.86505",
    "lng": "-87.60928"
  },
  {
    "idx": 310,
    "lat": "41.86404",
    "lng": "-87.60924"
  },
  {
    "idx": 320,
    "lat": "41.86225",
    "lng": "-87.60941"
  },
  {
    "idx": 330,
    "lat": "41.86059",
    "lng": "-87.61028"
  },
  {
    "idx": 340,
    "lat": "41.85949",
    "lng": "-87.60983"
  },
  {
    "idx": 350,
    "lat": "41.86020",
    "lng": "-87.61025"
  },
  {
    "idx": 360,
    "lat": "41.86193",
    "lng": "-87.60969"
  },
  {
    "idx": 370,
    "lat": "41.86330",
    "lng": "-87.60922"
  },
  {
    "idx": 380,
    "lat": "41.86482",
    "lng": "-87.60909"
  },
  {
    "idx": 390,
    "lat": "41.86586",
    "lng": "-87.61010"
  },
  {
    "idx": 400,
    "lat": "41.86647",
    "lng": "-87.61025"
  },
  {
    "idx": 410,
    "lat": "41.86625",
    "lng": "-87.61417"
  },
  {
    "idx": 420,
    "lat": "41.86504",
    "lng": "-87.61409"
  },
  {
    "idx": 430,
    "lat": "41.86321",
    "lng": "-87.61487"
  },
  {
    "idx": 440,
    "lat": "41.86177",
    "lng": "-87.61477"
  },
  {
    "idx": 450,
    "lat": "41.86028",
    "lng": "-87.61395"
  },
  {
    "idx": 460,
    "lat": "41.85861",
    "lng": "-87.61437"
  },
  {
    "idx": 470,
    "lat": "41.85818",
    "lng": "-87.61573"
  },
  {
    "idx": 480,
    "lat": "41.85642",
    "lng": "-87.61617"
  },
  {
    "idx": 490,
    "lat": "41.85184",
    "lng": "-87.61407"
  },
  {
    "idx": 500,
    "lat": "41.83902",
    "lng": "-87.60883"
  },
  {
    "idx": 510,
    "lat": "41.83400",
    "lng": "-87.60784"
  },
  {
    "idx": 520,
    "lat": "41.82918",
    "lng": "-87.60449"
  },
  {
    "idx": 530,
    "lat": "41.82518",
    "lng": "-87.60044"
  },
  {
    "idx": 540,
    "lat": "41.81778",
    "lng": "-87.59588"
  },
  {
    "idx": 550,
    "lat": "41.81264",
    "lng": "-87.59083"
  },
  {
    "idx": 560,
    "lat": "41.80761",
    "lng": "-87.58651"
  },
  {
    "idx": 570,
    "lat": "41.80389",
    "lng": "-87.58209"
  },
  {
    "idx": 580,
    "lat": "41.80025",
    "lng": "-87.58163"
  },
  {
    "idx": 590,
    "lat": "41.79966",
    "lng": "-87.58227"
  },
  {
    "idx": 600,
    "lat": "41.79947",
    "lng": "-87.58405"
  },
  {
    "idx": 610,
    "lat": "41.79560",
    "lng": "-87.58395"
  },
  {
    "idx": 620,
    "lat": "41.79342",
    "lng": "-87.58390"
  },
  {
    "idx": 630,
    "lat": "41.79253",
    "lng": "-87.58470"
  },
  {
    "idx": 640,
    "lat": "41.79194",
    "lng": "-87.58539"
  },
  {
    "idx": 650,
    "lat": "41.79032",
    "lng": "-87.58500"
  },
  {
    "idx": 660,
    "lat": "41.79047",
    "lng": "-87.58430"
  }
]]

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
        new_df.drop(columns=['Date', 'Time','YEAR','DAY'], inplace=True)
        new_df.rename(columns={0: 'LATITUDE', 1: 'LONGITUDE', 2: 'MONTH', 3: 'HOUR', 4: 'MINUTE'}, inplace=True)

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


#app.run(debug=True)


