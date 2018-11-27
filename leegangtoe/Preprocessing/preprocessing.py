import pandas as pd
import numpy as np

landmarks = pd.read_csv('landmarks.txt', sep = ',',header=None)

nose=landmarks.iloc[:][np.r_[54:72]]
nose.to_csv("nose_landmarks.txt",header=None,index=None, float_format='%.6f')

eyes=landmarks.iloc[:][np.r_[72:96]]
eyes.to_csv("eyes_landmarks.txt",header=None,index=None,float_format='%.6f')

lips=landmarks.iloc[:][np.r_[96:136]]
lips.to_csv("lips_landmarks.txt",header=None,index=None,float_format='%.6f')

face_conture=landmarks.iloc[:][np.r_[0:54]]
face_conture.to_csv("face_conture_landmarks.txt",header=None,index=None,float_format='%.6f')
