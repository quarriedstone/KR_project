import pandas as pd
from FeatureGeneration import generate_features

print("Eyes feature generation: start")
df_X = pd.read_csv('..\KR_project\DATASET\eyes_landmarks.txt', sep=',', header=None)
X = []
Y = []
arr = []
# Generating features from each row of coordinates
for i in range(int(df_X.shape[0])):
    for j in range(int(int(df_X.shape[1] / 2))):
        X.append(df_X.iloc[i][2 * j])
        Y.append(df_X.iloc[i][2 * j + 1])
    arr.append(generate_features(X, Y, df_X.shape))
    X.clear()
    Y.clear()
pd.DataFrame(arr).to_csv("eyes_landmarks_features.csv", index=False)
print("Eyes feature generation: end")


print("Nose feature generation: start")
df_X = pd.read_csv('..\KR_project\DATASET\\nose_landmarks.txt', sep=',', header=None)
X = []
Y = []
arr = []
# Generating features from each row of coordinates
for i in range(int(df_X.shape[0])):
    for j in range(int(int(df_X.shape[1] / 2))):
        X.append(df_X.iloc[i][2 * j])
        Y.append(df_X.iloc[i][2 * j + 1])
    arr.append(generate_features(X, Y, df_X.shape))
    X.clear()
    Y.clear()
pd.DataFrame(arr).to_csv("nose_landmarks_features.csv")
print("Nose feature generation: end")


print("Lips feature generation: start")
df_X = pd.read_csv('..\KR_project\DATASET\DATASET\lips_landmarks.txt', sep=',', header=None)
X = []
Y = []
arr = []
# Generating features from each row of coordinates
for i in range(int(df_X.shape[0])):
    for j in range(int(int(df_X.shape[1] / 2))):
        X.append(df_X.iloc[i][2 * j])
        Y.append(df_X.iloc[i][2 * j + 1])
    arr.append(generate_features(X, Y, df_X.shape))
    X.clear()
    Y.clear()
pd.DataFrame(arr).to_csv("lips_landmarks_features.csv")
print("Lips feature generation: end")

print("Face conture feature generation: start")
df_X = pd.read_csv('..\KR_project\DATASET\\face_conture_landmarks.txt', sep=',', header=None)
X = []
Y = []
arr = []
# Generating features from each row of coordinates
for i in range(int(df_X.shape[0])):
    for j in range(int(int(df_X.shape[1] / 2))):
        X.append(df_X.iloc[i][2 * j])
        Y.append(df_X.iloc[i][2 * j + 1])
    arr.append(generate_features(X, Y, df_X.shape))
    X.clear()
    Y.clear()
pd.DataFrame(arr).to_csv("face_conture_landmarks_features.csv")
print("Face conture feature generation: end")