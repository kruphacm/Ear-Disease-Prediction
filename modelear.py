import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20Ear%20-%20Sheet1.csv")
df['SYMPTOM 1'] = df['SYMPTOM 1'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Fluid drainage from the ear':9.0,'Painful':10.0,'Tender':11.0,'Red':12.0,"Swollen":13.0})

df['SYMPTOM 2'] = df['SYMPTOM 2'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Fluid drainage from the ear':9.0,'Painful':10.0,'Tender':11.0,'Red':12.0,"Swollen":13.0})

df['SYMPTOM 3'] = df['SYMPTOM 3'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Fluid drainage from the ear':9.0,'Painful':10.0,'Tender':11.0,'Red':12.0,"Swollen":13.0})

X = df.drop(['DISEASE','TREATMENT'], axis=1)
df['DISEASE'] = df['DISEASE'].map({'Ear Infection':0.0,'Inner ear Infection':1.0,'Middle Ear Infection':2.0,'Outer Ear Infection':3.0})
Y = df['DISEASE']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X,Y)
pickle.dump(model1, open('modelear.pkl','wb'))

model = pickle.load(open('modelear.pkl','rb'))