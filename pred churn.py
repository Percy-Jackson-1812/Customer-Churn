# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 04:05:13 2023

@author: ASUS
"""

import numpy as np
import pickle
load_model = pickle.load(open("C:/Users\ASUS/Downloads/best.sav", 'rb'))

i = (0.245,1,0.0002,0.501234,0.53753,1,0,0,0,0)
i_as_np = np.asarray(i)
i_re = i_as_np.reshape(-1,1)
p = load_model.predict(i_re)
print(p)
if(p==0):
    print("The employee didnt churn")
else:
    print("Employee churned")