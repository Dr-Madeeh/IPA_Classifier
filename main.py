# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:40:01 2021

@author: m_nay
"""

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
import pandas as pd
from nltk import word_tokenize
from nltk.stem.isri import ISRIStemmer
import re
import nltk
from stop_words import get_stop_words
import numpy as np

#!pip install stop_words
nltk.download('punkt')

def getsubstr(s,start,end): 
  return s[s.find(start)+len(start):s.rfind(end)]

from stop_words import get_stop_words
stop_words = get_stop_words('ar')

def remove_stopWords(s):
    '''For removing stop words
    '''
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s

sts = ISRIStemmer()
word_list = "عرض يستخدم الى التفاعل مع المستخدمين في هاذا المجال !وعلمآ تكون الخدمه للستطلاع على الخدمات والعروض المقدمة"

# Define a function
def filter(word_list):
    wordsfilter=[]
    for a in word_tokenize(word_list):
        stem = sts.stem(a)
        wordsfilter.append(stem)
    #print(wordsfilter)
    return ' '.join(wordsfilter)


df = pd.read_table(r'./123.txt')

modelname = 'modelPrimary.pickle'
vectorizername = 'vecPrimary.pickle'
clf = pickle.load(open(modelname, 'rb'))
vectorizer = pickle.load(open(vectorizername, 'rb'))



st.markdown("<h2 style='text-align: center;'>مصنف مركز البحوث والدراسات الآلي</h2>", unsafe_allow_html=True)
  

st.markdown("<h4 style='text-align: center; color: orange;'>الآن يمكنك تصنيف بياناتك طبقا للتصنيف المعتمد لدينا</h4>", unsafe_allow_html=True)


with st.form(key='mlform'):

    st.markdown("<h6 style='text-align: center;'>ادخل النص المراد تصنيفه</h6>", unsafe_allow_html=True)

    message = st.text_area("")
    submit_message = st.form_submit_button(label='صنف')
    
if submit_message:
    query = " ".join(re.findall('[\w]+',message))
    query = remove_stopWords(query)
    query = filter(query)
    
    predictions =clf.predict_proba(vectorizer .transform([query]))
    preds_idx = np.argsort(-predictions) 

    classes = pd.DataFrame(clf.classes_, columns=['class_name'])

    sum = 0
    nums = 0
    for i in range(10):
      if predictions[0][preds_idx[0][i]] < 0.1:
        break;
      else:
        nums = nums +1
        sum = sum + predictions[0][preds_idx[0][i]]
        #print(classes.iloc[preds_idx[0][i]])
        #print(predictions[0][preds_idx[0][i]])

    result = pd.DataFrame(columns=['predicted_class','predicted_prob'])

    for i in range(nums):
      #print(classes.iloc[preds_idx[0][i]])
      #print((predictions[0][preds_idx[0][i]]/sum)*100)
      s = getsubstr(str(classes.iloc[preds_idx[0][i]]),'class_name ','\n')
      dict = {'predicted_class': s, 'predicted_prob': (predictions[0][preds_idx[0][i]]/sum)*100}
      result = result.append(dict, ignore_index = True)

      st.markdown("<h3 style='text-align: center;color:red'>"+  s + " ("+ str(round((predictions[0][preds_idx[0][i]]/sum)*100),4) +"%)" +"</h3>", unsafe_allow_html=True)
      
        #pred = clf.predict(vectorizer.transform([message]))[0]
        #dd = df.loc[df['labelSecondary'] == pred]
        #dd = dd.iloc[[0]]
        #print(dd['label'])
        #st.title(pred)

        #st.markdown("<h3 style='text-align: center;color:red'>"+  pred +"</h3>", unsafe_allow_html=True)

    
