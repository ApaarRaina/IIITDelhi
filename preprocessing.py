import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial import cKDTree
from tqdm import tqdm

df=pd.read_csv('crime.csv')
l=np.array(df[' PRIMARY DESCRIPTION'])
l=np.array(set(l))


df.drop(columns=['CASE#',' IUCR',' SECONDARY DESCRIPTION','BEAT','FBI CD'],inplace=True)


df['DATE  OF OCCURRENCE'] = pd.to_datetime(df['DATE  OF OCCURRENCE'])


df['DATE'] = df['DATE  OF OCCURRENCE'].dt.date
df['TIME'] = df['DATE  OF OCCURRENCE'].dt.time

df.drop(columns=['DATE  OF OCCURRENCE'], inplace=True)

print(df.head())


df.dropna(inplace=True)


df['DATE']=pd.to_datetime(df['DATE'])

df['YEAR']=df['DATE'].dt.year
df['MONTH']=df['DATE'].dt.month
df['DAY']=df['DATE'].dt.day

df['HOUR']=df['TIME'].apply(lambda x: x.hour)
df['MINUTE']=df['TIME'].apply(lambda x: x.minute)
df.drop(columns=['DATE','TIME'],inplace=True)


# Define crime brutality scores
crime_brutality = {
    'OTHER OFFENSE': 3,
    'INTERFERENCE WITH PUBLIC OFFICER': 2,
    'CRIMINAL DAMAGE': 3,
    'PROSTITUTION': 4,
    'WEAPONS VIOLATION': 10,  # Increased
    'LIQUOR LAW VIOLATION': 1,
    'PUBLIC INDECENCY': 2,
    'OFFENSE INVOLVING CHILDREN': 20,  # Increased
    'BURGLARY': 9,
    'HOMICIDE': 25,  # Increased significantly
    'ARSON': 12,  # Increased
    'GAMBLING': 4,
    'CONCEALED CARRY LICENSE VIOLATION': 8,
    'PUBLIC PEACE VIOLATION': 5,
    'ROBBERY': 10,  # Increased
    'STALKING': 10,
    'OBSCENITY': 5,
    'THEFT': 5,  # Decreased
    'ASSAULT': 8,  # Decreased
    'NON-CRIMINAL': 1,
    'CRIMINAL TRESPASS': 5,
    'DECEPTIVE PRACTICE': 4,
    'OTHER NARCOTIC VIOLATION': 7,
    'KIDNAPPING': 22,  # Increased
    'BATTERY': 7,  # Decreased
    'SEX OFFENSE': 18,  # Increased
    'NARCOTICS': 5,
    'INTIMIDATION': 8,
    'CRIMINAL SEXUAL ASSAULT': 25,  # Increased significantly
    'MOTOR VEHICLE THEFT': 6,
    'HUMAN TRAFFICKING': 22,  # Increased
    'USER BASED': 5
}

# Map brutality scores to the dataframe
df['Brutality_Score'] = df[' PRIMARY DESCRIPTION'].map(crime_brutality).fillna(1)

# Ensure DATE column is in datetime format
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']])

# Extract coordinates
coords = df[['X COORDINATE', 'Y COORDINATE']].values

# Build KDTree
tree = cKDTree(coords)

# Find all nearby crime indices at once (reduces redundant queries)
all_nearby_indices = tree.query_ball_point(coords, 50)

# Convert DATE to NumPy array for fast comparisons
dates = df['DATE'].to_numpy()
hours = df['HOUR'].to_numpy()
scores = df['Brutality_Score'].to_numpy()


# Vectorized function to compute crime severity
def compute_severity(idx):
    nearby_idx = all_nearby_indices[idx]

    # Get nearby crimes data
    nearby_dates = dates[nearby_idx]
    nearby_hours = hours[nearby_idx]
    nearby_scores = scores[nearby_idx]

    # Time filtering (past 30 days and ±2 hours)
    valid_mask = (nearby_dates >= dates[idx] - np.timedelta64(30, 'D')) & \
                 (nearby_dates <= dates[idx]) & \
                 (np.abs(nearby_hours - hours[idx]) <= 2)

    return nearby_scores[valid_mask].sum()


# Compute severity scores using NumPy for fast looping
df['Crime_Severity_Score'] = np.array([compute_severity(i) for i in range(len(df))])

print(df['Crime_Severity_Score'].describe())


score=np.array(df['Crime_Severity_Score'])

print(score)
print(np.bincount(score))

df.loc[(df['Crime_Severity_Score'] >= 0) & (df['Crime_Severity_Score'] < 10), 'Crime_Severity_Label'] = 0
df.loc[(df['Crime_Severity_Score'] >= 10) & (df['Crime_Severity_Score'] < 30), 'Crime_Severity_Label'] = 1
df.loc[(df['Crime_Severity_Score'] >= 30) & (df['Crime_Severity_Score'] < 70), 'Crime_Severity_Label'] = 2
df.loc[(df['Crime_Severity_Score'] >= 70) & (df['Crime_Severity_Score'] < 90), 'Crime_Severity_Label'] = 3
df.loc[(df['Crime_Severity_Score'] >= 90) & (df['Crime_Severity_Score'] < 120), 'Crime_Severity_Label'] = 4
df.loc[(df['Crime_Severity_Score'] >= 120) & (df['Crime_Severity_Score'] < 100), 'Crime_Severity_Label'] = 5
df.loc[(df['Crime_Severity_Score'] >= 100) & (df['Crime_Severity_Score'] < 130), 'Crime_Severity_Label'] = 6
df.loc[(df['Crime_Severity_Score'] >= 130) & (df['Crime_Severity_Score'] < 150), 'Crime_Severity_Label'] = 7
df.loc[(df['Crime_Severity_Score'] >= 150) & (df['Crime_Severity_Score'] < 180), 'Crime_Severity_Label'] = 8
df.loc[(df['Crime_Severity_Score'] >= 180), 'Crime_Severity_Label'] = 9

df.drop(columns=['BLOCK',' PRIMARY DESCRIPTION',' LOCATION DESCRIPTION','ARREST','DOMESTIC','WARD','LOCATION','X COORDINATE','Y COORDINATE','Brutality_Score','Crime_Severity_Score'],inplace=True)

'''df['DATE']=pd.to_datetime(df['DATE'])

df['YEAR']=df['DATE'].dt.year
df['MONTH']=df['DATE'].dt.month
df['DAY']=df['DATE'].dt.day

df['HOUR']=df['TIME'].apply(lambda x: x.hour)
df['MINUTE']=df['TIME'].apply(lambda x: x.minute)

df['SAFETY']=df['Crime_Severity_Label']

df.drop(columns=['Crime_Severity_Label','DATE','TIME'],inplace=True)
df'''



print(df['Crime_Severity_Label'].describe())
df.drop(columns=['DATE'],inplace=True)

corr_matrix=df.corr()
print(corr_matrix)

df.drop(columns=['YEAR'],inplace=True)

df.drop(columns=['DAY'],inplace=True)

# Morning: 7 AM to 11:59 AM
df.loc[(df['HOUR'] > 6) & (df['HOUR'] < 12), 'HOUR'] = 0

# Afternoon: 12 PM to 3:59 PM
df.loc[(df['HOUR'] >= 12) & (df['HOUR'] < 16), 'HOUR'] = 1

# Evening: 4 PM to 6:59 PM
df.loc[(df['HOUR'] >= 16) & (df['HOUR'] < 19), 'HOUR'] = 2

# Night: 7 PM to 5:59 AM
df.loc[(df['HOUR'] >= 19) | (df['HOUR'] < 6), 'HOUR'] = 3


df.to_csv('Processed_crime.csv')