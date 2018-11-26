import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_X = pd.read_csv('SCUT-FBP-1.csv', sep = ',')
X=[]
Y=[]
Z=[]
for i in range(67):
    X.append(df_X.iloc[0][' x_'+str(i)])
    Y.append(df_X.iloc[0][' y_'+str(i)])
    Z.append(i)
print(X)
print(Y)

fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(Z):
    ax.annotate(txt, (X[i], Y[i]))
plt.show()
