import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20Ear%20-%20Sheet1.csv")
df['SYMPTOM 1'] = df['SYMPTOM 1'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Fluid drainage from the ear':11.0,'Painful':12.0,"Tender":13.0,'Red':14.0,"Swollen":15.0})

df['SYMPTOM 2'] = df['SYMPTOM 2'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Fluid drainage from the ear':11.0,'Painful':12.0,"Tender":13.0,'Red':14.0,"Swollen":15.0})

df['SYMPTOM 3'] = df['SYMPTOM 3'].map({'Mild pain or Discomfort inside the ear':0.0,'A feeling of Pressure inside the ear':1.0,'Pus':2.0,'Hearing loss':3.0,'Dizziness':4.0,'Nausea':5.0,'Vomiting':6.0,'Eardrum Bulge':7.0,'Ear ache':8.0,'Discolored nail that are brown,yellow,white':9.0,'Fragile and cracked nail':10.0,'Fluid drainage from the ear':11.0,'Painful':12.0,"Tender":13.0,'Red':14.0,"Swollen":15.0})

X = df.drop(['DISEASE','TREATMENT'], axis=1)
df['DISEASE'] = df['DISEASE'].map({'Ear Infection':0.0,'Inner ear Infection':1.0,'Middle Ear Infection':2.0,'Outer Ear Infection':3.0})
Y = df['DISEASE']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

model1=LogisticRegression()
model1.fit(X,Y)
pickle.dump(model1, open('modelear.pkl','wb'))

model = pickle.load(open('modelear.pkl','rb'))