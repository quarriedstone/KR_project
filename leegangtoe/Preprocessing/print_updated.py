import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

array=[]
for i in range(2000):
    if (i+1==1112):
        df_X = pd.read_csv('./processed/AF'+str(i)+'.csv', sep = ',')
    else:
        df_X = pd.read_csv('./processed/AF'+str(i+1)+'.csv', sep = ',')
    inner_array=[]
    for ii in range(68):
        inner_array.append(df_X.iloc[0][' x_'+str(ii)])
        inner_array.append(df_X.iloc[0][' y_'+str(ii)])
    print(len(inner_array))
    array.append(inner_array)
np.array(array)
        
np.savetxt('./landmarks.txt', array, delimiter=',', fmt = '%.06f')