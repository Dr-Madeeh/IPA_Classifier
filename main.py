# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:40:01 2021

@author: m_nay
"""

import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
import pandas as pd


df = pd.read_table(r'./123.txt')

modelname = 'model.pickle'
vectorizername = 'vec.pickle'
clf = pickle.load(open(modelname, 'rb'))
vectorizer = pickle.load(open(vectorizername, 'rb'))



st.markdown("<h2 style='text-align: center;'>مصنف مركز البحوث والدراسات الآلي</h2>", unsafe_allow_html=True)
  

st.markdown("<h4 style='text-align: center; color: orange;'>الآن يمكنك تصنيف بياناتك طبقا للتصنيف المعتمد لدينا</h4>", unsafe_allow_html=True)


with st.form(key='mlform'):

    st.markdown("<h6 style='text-align: center;'>ادخل النص المراد تصنيفه</h6>", unsafe_allow_html=True)

    message = st.text_area("")
    submit_message = st.form_submit_button(label='صنف')
    
if submit_message:
    pred = clf.predict(vectorizer.transform([message]))[0]
    dd = df.loc[df['labelSecondary'] == pred]
    dd = dd.iloc[[0]]
    #print(dd['label'])
    #st.title(pred)

    st.markdown("<h3 style='text-align: center;color:red'>"+  pred +"</h3>", unsafe_allow_html=True)

    