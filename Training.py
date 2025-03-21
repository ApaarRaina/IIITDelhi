import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv('processed_crime.csv')

scaler=StandardScaler()


train,test=train_test_split(df,test_size=0.1)
X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]
X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]


X_train.drop(columns=['Unnamed: 0'],inplace=True)
X_test.drop(columns=['Unnamed: 0'],inplace=True)


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

# Define XGBoost model
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",  # Multiclass classification
    num_class=10,  # Number of classes (0 to 4)
    n_estimators=250,  # Number of trees
    max_depth=10,  # Depth of trees
    learning_rate=0.1,  # Step size
    subsample=0.7,  # Avoid overfitting
    colsample_bytree=0.7,
    eval_metric="merror",
    random_state=42
)


# Train the model
xgb_clf.fit(X_train, y_train)

# Make predictions

y_pred_test = xgb_clf.predict(X_test)

# Calculate accuracy


acc = accuracy_score(y_test, y_pred_test)
print(f"XGBoost Accuracy: Test Accuracy {acc:.4f}")
y_pred=xgb_clf.predict(X_train)
acc=accuracy_score(y_train,y_pred)
print(f"XGBoost Accuracy: Train Accuracy {acc:.4f}")


print(np.bincount(y_pred_test,minlength=10))
print(np.bincount(y_test,minlength=10))


route=[
  {
    "idx": 0,
    "lat": "41.88272",
    "lng": "-87.62410"
  },
  {
    "idx": 10,
    "lat": "41.88208",
    "lng": "-87.62593"
  },
  {
    "idx": 20,
    "lat": "41.88215",
    "lng": "-87.62782"
  },
  {
    "idx": 30,
    "lat": "41.88437",
    "lng": "-87.62790"
  },
  {
    "idx": 40,
    "lat": "41.88576",
    "lng": "-87.62794"
  },
  {
    "idx": 50,
    "lat": "41.88681",
    "lng": "-87.62738"
  },
  {
    "idx": 60,
    "lat": "41.88715",
    "lng": "-87.62633"
  },
  {
    "idx": 70,
    "lat": "41.88823",
    "lng": "-87.62514"
  },
  {
    "idx": 80,
    "lat": "41.88820",
    "lng": "-87.62366"
  },
  {
    "idx": 90,
    "lat": "41.88784",
    "lng": "-87.62090"
  },
  {
    "idx": 100,
    "lat": "41.88756",
    "lng": "-87.61900"
  },
  {
    "idx": 110,
    "lat": "41.88682",
    "lng": "-87.61969"
  },
  {
    "idx": 120,
    "lat": "41.88670",
    "lng": "-87.62051"
  },
  {
    "idx": 130,
    "lat": "41.88686",
    "lng": "-87.62094"
  },
  {
    "idx": 140,
    "lat": "41.88680",
    "lng": "-87.62067"
  },
  {
    "idx": 150,
    "lat": "41.88856",
    "lng": "-87.62048"
  },
  {
    "idx": 160,
    "lat": "41.89070",
    "lng": "-87.62023"
  },
  {
    "idx": 170,
    "lat": "41.89180",
    "lng": "-87.62024"
  },
  {
    "idx": 180,
    "lat": "41.89174",
    "lng": "-87.62378"
  }
]


Time=["2024-06-15", "2:30:00"]

location_list=[]


for i in range(len(route)):
    l=[]
    for key in route[i]:
      if key=='lat' or key=='lng':
        l.append(float(route[i][key]))

    location_list.append(l)

new_df=pd.DataFrame(location_list)
new_df['Date']=Time[0]
new_df['Time']=Time[1]
new_df['Date']=pd.to_datetime(new_df['Date'])
new_df['Time']=pd.to_datetime(new_df['Time'])

new_df['YEAR']=new_df['Date'].dt.year
new_df[2]=new_df['Date'].dt.month
new_df['DAY']=new_df['Date'].dt.day

new_df[3]=new_df['Time'].apply(lambda x: x.hour)
new_df[4]=new_df['Time'].apply(lambda x: x.minute)
new_df.drop(columns=['Date','Time','YEAR','DAY'],inplace=True)

print(new_df)

new_df_values=scaler.transform(new_df)

y_pred=xgb_clf.predict(new_df_values)

print(y_pred)

sum=np.sum(y_pred)

div=len(y_pred)*10

score=sum/div

print(score)

with open('model.pkl','wb') as f:
  pickle.dump(xgb_clf,f)


counts = np.bincount(y_pred, minlength=10)

print(counts)

