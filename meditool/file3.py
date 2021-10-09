import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import datetime
from datetime import datetime, date, time
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('training_data.csv')
df_train.drop(['Unnamed: 133'],axis=1,inplace=True)
df_test = pd.read_csv('test_data.csv')
df = pd.concat([df_train,df_test],ignore_index=True)

y = df['prognosis']
X = df.drop(['prognosis'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=54, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

y_pred_train_gini = clf_gini.predict(X_train)


y_real_train_gini = y_train.values

st.title("Welcome to MediTool",anchor=None)
st.sidebar.title("Enter your symptoms")
st.subheader("Best health care consultant application",anchor=None)


st.date_input('Date When the Synptoms Started')

Location = ["Tambaram","Egmore","Guindy"]
loc = st.selectbox('Location',Location)
st.write("--------------------")
symptom1 = st.sidebar.selectbox('Symptom1',X_train.columns)
symptom2 = st.sidebar.selectbox('Symptom2',X_train.columns)
symptom3 = st.sidebar.selectbox('Symptom3',X_train.columns)
symptom4 = st.sidebar.selectbox('Symptom4',X_train.columns)
symptom5 = st.sidebar.selectbox('Symptom5',X_train.columns)
symptom6 = st.sidebar.selectbox('Symptom6',X_train.columns)
symptom7 = st.sidebar.selectbox('Symptom7',X_train.columns)
symptom8 = st.sidebar.selectbox('Symptom8',X_train.columns)
symptom9 = st.sidebar.selectbox('Symptom9',X_train.columns)
symptom10 = st.sidebar.selectbox('Symptom10',X_train.columns)
symptom11 = st.sidebar.selectbox('Symptom11',X_train.columns)

submit = st.sidebar.button ("SUBMIT")

if submit:
    ds = [0]*132
    a = list(X_train.columns)
    for i in range(len(a)):
        if a[i] == symptom1:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom2:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom3:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom4:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom5:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom6:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom7:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom8:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom9:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom10:
            ds[i]= 1
    for i in range(len(a)):
        if a[i] == symptom11:
            ds[i]= 1
    sym = np.array(ds)
    sym = sym.reshape(1,-1)
    y_pred_train_en = clf_gini.predict(sym)
    ss = np.array_str(y_pred_train_en)
    ss = ss.replace("[","").replace("]","")
    st.header("Predicted Disease From Symptoms")
    st.header(ss)
    
    if loc == "Tambaram":
        st.write("--------------------")
        st.subheader("2 HOSPITALS FOUND")
        st.write("--------------------")
        st.write("Hindu Mission Hospital")
        st.write("Address: 103, Grand Southern Trunk Rd, New Market, Tambaram West")
        st.write("Tambaram, Chennai, Tamil Nadu 600045")
        st.write("[Direction](https://goo.gl/maps/Ls2mYpSLBGynTPF89)")
        st.write("--------------------")
        st.write("Kasthuri Hospital")
        st.write("Address: 119, Shanmugam Road, West Tambaram")
        st.write("Chennai, Tamil Nadu 600045")
        st.write("[Direction](https://g.page/kasthuri-hospital?share)")

    if loc == "Egmore":
        st.write("--------------------")
        st.subheader("2 HOSPITALS FOUND")
        st.write("--------------------")
        st.write("Government Hospital Egmore")
        st.write("Address: Pantheon Rd, Komaleeswaranpet")
        st.write("Egmore, Chennai, Tamil Nadu 600008")
        st.write("[Direction](https://goo.gl/maps/bZHzrUhNUz9yGqiw8)")
        st.write("--------------------")
        st.write("Westminster Hospitals")
        st.write("Address: New No 2, 145, Nungambakkam High Rd, opp. The Park Hotel, Thousand Lights West")
        st.write("Thousand Lights, Chennai, Tamil Nadu 600034")
        st.write("[Direction](https://g.page/WestminsterHospitals?share)")

    if loc == "Guindy":
        st.write("--------------------")
        st.subheader("2 HOSPITALS FOUND")
        st.write("--------------------")
        st.write("Apollo Health And Lifestyle Limited")
        st.write("1st floor, Sardar Patel Rd, Guindy National Park")
        st.write("Guindy, Chennai, Tamil Nadu 600022")
        st.write("[Direction](https://goo.gl/maps/WYyp7zHwXAwYZuYv9)")
        st.write("--------------------")
        st.write("St Thomas Hospital")
        st.write("5, 105, Defence Colony 1st Ave, Seven Wells")
        st.write("St.Thomas Mount, Tamil Nadu 600016")
        st.write("[Direction](https://goo.gl/maps/g7bChvEN7MSw1shB7)")



