import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle

# Load dataset
df = pd.read_csv('processed_crime.csv')


# Split into train and test sets
train, test = train_test_split(df, test_size=0.1, random_state=42)
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

# Drop unnecessary column
X_train.drop(columns=['Unnamed: 0'], inplace=True)
X_test.drop(columns=['Unnamed: 0'], inplace=True)


# Define XGBoost model with sample weights
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",  # Multiclass classification
    num_class=10,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="merror",
    random_state=42
)

xgb_clf.fit(X_train, y_train)


y_pred_test = xgb_clf.predict(X_test)
y_pred_train = xgb_clf.predict(X_train)

# Calculate accuracy
test_acc = accuracy_score(y_test, y_pred_test)
train_acc = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy {train_acc:.4f}")
print(f"Test Accuracy {test_acc:.4f}")


# Route data for new predictions
route = [
  {
    "idx": 0,
    "lat": "41.874155",
    "lng": "-87.620593"
  },
  {
    "idx": 1,
    "lat": "41.874521",
    "lng": "-87.620602"
  },
  {
    "idx": 2,
    "lat": "41.875053",
    "lng": "-87.620616"
  },
  {
    "idx": 3,
    "lat": "41.875388",
    "lng": "-87.620626"
  },
  {
    "idx": 4,
    "lat": "41.875625",
    "lng": "-87.620633"
  },
  {
    "idx": 5,
    "lat": "41.875813",
    "lng": "-87.620635"
  },
  {
    "idx": 6,
    "lat": "41.875888",
    "lng": "-87.620636"
  },
  {
    "idx": 7,
    "lat": "41.875957",
    "lng": "-87.620637"
  },
  {
    "idx": 8,
    "lat": "41.876287",
    "lng": "-87.620643"
  },
  {
    "idx": 9,
    "lat": "41.877356",
    "lng": "-87.620671"
  },
  {
    "idx": 10,
    "lat": "41.877905",
    "lng": "-87.620697"
  },
  {
    "idx": 11,
    "lat": "41.878226",
    "lng": "-87.620701"
  },
  {
    "idx": 12,
    "lat": "41.87833",
    "lng": "-87.620707"
  },
  {
    "idx": 13,
    "lat": "41.878453",
    "lng": "-87.62072"
  },
  {
    "idx": 14,
    "lat": "41.879734",
    "lng": "-87.620751"
  },
  {
    "idx": 15,
    "lat": "41.880602",
    "lng": "-87.620782"
  },
  {
    "idx": 16,
    "lat": "41.880752",
    "lng": "-87.620788"
  },
  {
    "idx": 17,
    "lat": "41.880871",
    "lng": "-87.620794"
  },
  {
    "idx": 18,
    "lat": "41.880874",
    "lng": "-87.620598"
  },
  {
    "idx": 19,
    "lat": "41.88088",
    "lng": "-87.620184"
  },
  {
    "idx": 20,
    "lat": "41.880893",
    "lng": "-87.619331"
  },
  {
    "idx": 21,
    "lat": "41.880897",
    "lng": "-87.619061"
  },
  {
    "idx": 22,
    "lat": "41.880901",
    "lng": "-87.61876"
  },
  {
    "idx": 23,
    "lat": "41.880917",
    "lng": "-87.617692"
  },
  {
    "idx": 24,
    "lat": "41.880918",
    "lng": "-87.617664"
  },
  {
    "idx": 25,
    "lat": "41.88092",
    "lng": "-87.617506"
  },
  {
    "idx": 26,
    "lat": "41.880924",
    "lng": "-87.617244"
  },
  {
    "idx": 27,
    "lat": "41.881044",
    "lng": "-87.617247"
  },
  {
    "idx": 28,
    "lat": "41.881379",
    "lng": "-87.617238"
  },
  {
    "idx": 29,
    "lat": "41.881567",
    "lng": "-87.617218"
  },
  {
    "idx": 30,
    "lat": "41.881717",
    "lng": "-87.617191"
  },
  {
    "idx": 31,
    "lat": "41.881826",
    "lng": "-87.617167"
  },
  {
    "idx": 32,
    "lat": "41.881929",
    "lng": "-87.617128"
  },
  {
    "idx": 33,
    "lat": "41.882083",
    "lng": "-87.617061"
  },
  {
    "idx": 34,
    "lat": "41.882205",
    "lng": "-87.616996"
  },
  {
    "idx": 35,
    "lat": "41.882262",
    "lng": "-87.616957"
  },
  {
    "idx": 36,
    "lat": "41.882445",
    "lng": "-87.616825"
  },
  {
    "idx": 37,
    "lat": "41.882559",
    "lng": "-87.616723"
  },
  {
    "idx": 38,
    "lat": "41.882664",
    "lng": "-87.616611"
  },
  {
    "idx": 39,
    "lat": "41.882764",
    "lng": "-87.616463"
  },
  {
    "idx": 40,
    "lat": "41.882876",
    "lng": "-87.616282"
  },
  {
    "idx": 41,
    "lat": "41.882937",
    "lng": "-87.616155"
  },
  {
    "idx": 42,
    "lat": "41.883022",
    "lng": "-87.615967"
  },
  {
    "idx": 43,
    "lat": "41.883069",
    "lng": "-87.615843"
  },
  {
    "idx": 44,
    "lat": "41.88315",
    "lng": "-87.615614"
  },
  {
    "idx": 45,
    "lat": "41.883254",
    "lng": "-87.615316"
  },
  {
    "idx": 46,
    "lat": "41.88334",
    "lng": "-87.615072"
  },
  {
    "idx": 47,
    "lat": "41.883463",
    "lng": "-87.61479"
  },
  {
    "idx": 48,
    "lat": "41.883561",
    "lng": "-87.6146"
  },
  {
    "idx": 49,
    "lat": "41.883639",
    "lng": "-87.614476"
  },
  {
    "idx": 50,
    "lat": "41.883727",
    "lng": "-87.614349"
  },
  {
    "idx": 51,
    "lat": "41.883849",
    "lng": "-87.614208"
  },
  {
    "idx": 52,
    "lat": "41.883982",
    "lng": "-87.614076"
  },
  {
    "idx": 53,
    "lat": "41.8841",
    "lng": "-87.613977"
  },
  {
    "idx": 54,
    "lat": "41.884205",
    "lng": "-87.613901"
  },
  {
    "idx": 55,
    "lat": "41.884344",
    "lng": "-87.61381"
  },
  {
    "idx": 56,
    "lat": "41.884512",
    "lng": "-87.613723"
  },
  {
    "idx": 57,
    "lat": "41.88464",
    "lng": "-87.613682"
  },
  {
    "idx": 58,
    "lat": "41.884762",
    "lng": "-87.613645"
  },
  {
    "idx": 59,
    "lat": "41.884932",
    "lng": "-87.613614"
  },
  {
    "idx": 60,
    "lat": "41.885068",
    "lng": "-87.613605"
  },
  {
    "idx": 61,
    "lat": "41.885234",
    "lng": "-87.613606"
  },
  {
    "idx": 62,
    "lat": "41.885476",
    "lng": "-87.613633"
  },
  {
    "idx": 63,
    "lat": "41.885835",
    "lng": "-87.613688"
  },
  {
    "idx": 64,
    "lat": "41.886113",
    "lng": "-87.613736"
  },
  {
    "idx": 65,
    "lat": "41.886393",
    "lng": "-87.613776"
  },
  {
    "idx": 66,
    "lat": "41.887363",
    "lng": "-87.613914"
  },
  {
    "idx": 67,
    "lat": "41.887566",
    "lng": "-87.613932"
  },
  {
    "idx": 68,
    "lat": "41.887886",
    "lng": "-87.613973"
  },
  {
    "idx": 69,
    "lat": "41.887956",
    "lng": "-87.613975"
  },
  {
    "idx": 70,
    "lat": "41.888908",
    "lng": "-87.614001"
  },
  {
    "idx": 71,
    "lat": "41.888976",
    "lng": "-87.614003"
  },
  {
    "idx": 72,
    "lat": "41.889097",
    "lng": "-87.614006"
  },
  {
    "idx": 73,
    "lat": "41.889391",
    "lng": "-87.61389"
  },
  {
    "idx": 74,
    "lat": "41.889546",
    "lng": "-87.613857"
  },
  {
    "idx": 75,
    "lat": "41.890139",
    "lng": "-87.613758"
  },
  {
    "idx": 76,
    "lat": "41.890245",
    "lng": "-87.613737"
  },
  {
    "idx": 77,
    "lat": "41.890398",
    "lng": "-87.613694"
  },
  {
    "idx": 78,
    "lat": "41.89056",
    "lng": "-87.613604"
  },
  {
    "idx": 79,
    "lat": "41.890692",
    "lng": "-87.613529"
  },
  {
    "idx": 80,
    "lat": "41.890916",
    "lng": "-87.613477"
  },
  {
    "idx": 81,
    "lat": "41.891067",
    "lng": "-87.613421"
  },
  {
    "idx": 82,
    "lat": "41.891235",
    "lng": "-87.613381"
  },
  {
    "idx": 83,
    "lat": "41.89144",
    "lng": "-87.613331"
  },
  {
    "idx": 84,
    "lat": "41.891692",
    "lng": "-87.613258"
  },
  {
    "idx": 85,
    "lat": "41.891956",
    "lng": "-87.613174"
  },
  {
    "idx": 86,
    "lat": "41.892228",
    "lng": "-87.613081"
  },
  {
    "idx": 87,
    "lat": "41.892483",
    "lng": "-87.61301"
  },
  {
    "idx": 88,
    "lat": "41.892719",
    "lng": "-87.61295"
  },
  {
    "idx": 89,
    "lat": "41.892993",
    "lng": "-87.61291"
  }
]


Time = ["2024-06-15", "2:30:00"]

# Convert route data to DataFrame
location_list = [[float(point['lat']), float(point['lng'])] for point in route]

new_df = pd.DataFrame(location_list)
new_df['Date'] = pd.to_datetime(Time[0])
new_df['Time'] = pd.to_datetime(Time[1])

# Extract time features
new_df['YEAR'] = new_df['Date'].dt.year
new_df[2] = new_df['Date'].dt.month
new_df['DAY'] = new_df['Date'].dt.day
new_df[3] = new_df['Time'].apply(lambda x: x.hour)
new_df[4] = new_df['Time'].apply(lambda x: x.minute)

# Drop unnecessary columns
new_df.drop(columns=['Date', 'Time','YEAR','DAY'], inplace=True)

new_df.rename(columns={0:'LATITUDE',1:'LONGITUDE',2:'MONTH',3:'HOUR',4:'MINUTE'},inplace=True)


# Make predictions
y_pred = xgb_clf.predict(new_df)

# Compute crime risk score
score = np.sum(y_pred) / (len(y_pred))

print(f"Crime Risk Score: {score:.4f}")

with open('model.pkl','wb') as f:
  pickle.dump(xgb_clf,f)


