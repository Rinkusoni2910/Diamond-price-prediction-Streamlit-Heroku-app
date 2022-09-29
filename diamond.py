
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
df = pd.read_csv('./diamonds.csv')

import pickle
from pickle import load
scaler = load(open('models/standard_scaler.pkl', 'rb'))
le = load(open('models/le.pkl', 'rb'))
knn_regressor=load(open('models/knn_model.pkl', 'rb'))

st.title("Know your Diamond")

st.subheader('Anatomy of Diamond')
img = Image.open("pic/d.png") 
st.image(img, width=700)
st.subheader('Quality of the Cut')
img = Image.open("pic/e.jfif") 
st.image(img, width=700)
st.subheader('Color of the Diamond')
img = Image.open("pic/f.jfif") 
st.image(img, width=700)
st.subheader('Clarity of the Diamond')
img = Image.open("pic/g.jfif") 
st.image(img, width=700)
st.title("Predict the diamond price")
st.subheader("Enter details of diamond below to predict it`s price")
with st.form('my_form'):
    carat = st.text_input("Weight")
    cut = st.text_input("Quality of the cut")
    color = st.text_input("Color")
    clarity = st.text_input("Clarity")
    x = st.text_input("Length(mm)")
    y = st.text_input("Width(mm)")
    z = st.text_input("Depth(mm)")
    depth = st.text_input("Total depth(%)")
    table = st.text_input("width of top of diamond relative to widest point")

    btn=st.form_submit_button(label='Predict')
if btn:
    if carat and cut and color and clarity and x and y and z and depth and table:
        label_cut = {'Ideal':2, 'Premium':3, 'Very Good':4, 'Good':1, 'Fair':0}
        label_color = {'G':3, 'E':1, 'F':2, 'H':4, 'D':0, 'I':5, 'J':6}
        label_clarity = {'SI1':2, 'VS2':5, 'SI2':3, 'VS1': 4, 'VVS2':7, 'VVS1':6, 'IF':1, 'I1':0}

        cut1= label_cut[cut]
        color1= label_color[color]
        clarity1= label_clarity[clarity]

        query_point = np.array([carat,cut1,color1,clarity1,x,y,z,depth,table])
        query_point = query_point.reshape(1, -1)

        query_point_transformed = scaler.transform(query_point)
        price=knn_regressor.predict(query_point_transformed)
        st.success(f"The price of Selected Diamond is $ {round(price[0],3)}")
else:
    st.error('Please Enter all the values') 