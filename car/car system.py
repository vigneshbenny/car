# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:19:00 2024

@author: LENOVO
"""

import numpy as np
import joblib
C=joblib.load("C:/Users/LENOVO/Downloads/archive (3)/car_price.joblib")
print("check your car price")
input_data=(3,1,1,0,2,0,8.6,258,0,130,3.47,2.68,111,5000,21,27,)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
result=C.predict(input_data_reshaped)
print(result)
