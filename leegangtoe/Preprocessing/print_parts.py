import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_X = pd.read_csv('face_conture_landmarks.txt', sep = ',',header=None)
X=[]
Y=[]
Z=[]
print(df_X.shape)
for i in range(int(int(df_X.shape[1])/2)):
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

df_X = pd.read_csv('nose_landmarks.txt', sep = ',',header=None)
X=[]
Y=[]
Z=[]
print(df_X.shape)
for i in range(int(int(df_X.shape[1])/2)):
    X.append(df_X.iloc[0][2*i])
    Y.append(df_X.iloc[0][2*i+1])
    Z.append(27+i)
print(X)
print(Y)

fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(Z):
    ax.annotate(txt, (X[i], Y[i]))
plt.show()

df_X = pd.read_csv('eyes_landmarks.txt', sep = ',',header=None)
X=[]
Y=[]
Z=[]
print(df_X.shape)
for i in range(int(int(df_X.shape[1])/2)):
    X.append(df_X.iloc[0][2*i])
    Y.append(df_X.iloc[0][2*i+1])
    Z.append(36+i)
print(X)
print(Y)

fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(Z):
    ax.annotate(txt, (X[i], Y[i]))
plt.show()

df_X = pd.read_csv('lips_landmarks.txt', sep = ',',header=None)
X=[]
Y=[]
Z=[]
print(df_X.shape)
for i in range(int(int(df_X.shape[1])/2)):
    X.append(df_X.iloc[0][2*i])
    Y.append(df_X.iloc[0][2*i+1])
    Z.append(48+i)
print(X)
print(Y)

fig, ax = plt.subplots()
ax.scatter(X, Y)

for i, txt in enumerate(Z):
    ax.annotate(txt, (X[i], Y[i]))
plt.show()
