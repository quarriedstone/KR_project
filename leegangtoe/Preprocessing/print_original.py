import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_X = pd.read_csv('./landmarks.txt', sep = ',',header=None)
X=[]
Y=[]
Z=[]
for i in range(67):
    X.append(df_X.iloc[0][2*i])
    Y.append(df_X.iloc[0][2*i+1])
    Z.append(i)
print(X)
print(Y)

fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(Z):
    ax.annotate(txt, (X[i], Y[i]))
plt.show()
