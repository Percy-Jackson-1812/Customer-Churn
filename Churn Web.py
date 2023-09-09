# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 23:21:20 2023

@author: Percy
"""

import numpy as np
import pickle
import streamlit as st
load_model = pickle.load(open("C:/Users/ASUS/Downloads/best.sav", 'rb'))
def pred(i):
    #i = (0.245,1,0.0002,0.501234,0.53753,1,0,0,0,0)
    i = np.array([i]).astype(np.float64)
    i_as_np = np.asarray(i)
    i_re = i_as_np.reshape(-1,1)
    p = load_model.predict(i_re)
    print(p)
    if(p==0):
        print("The employee didnt churn")
    else:
        print("Employee churned")
def main():
    st.title('Churn Model')
    age = st.text_input("Age")
    g = st.text_input("Gender")
    sub = st.text_input("Subscription")
    mon = st.text_input("Monthly Bills")
    gb = st.text_input("Total Usage")
    chi = st.text_input("Chicago")
    h = st.text_input("Houston")
    la = st.text_input("LA")
    mi = st.text_input("Miami")
    ny = st.text_input("New York")
    
    ans = ''
    if st.button("Answer"):
        
        ans = pred([age,g,sub,mon,gb,chi,h,la,mi,ny])
    st.success(ans)
    
    
if __name__=='__main__':
    main()
