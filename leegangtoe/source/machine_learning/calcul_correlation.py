import numpy as np

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process

#calculating pearson correlation
ratings = np.loadtxt('../../data/ratings.txt', delimiter=',')
base='../../results/cross_valid_predictions_'
target='lasso_100'
#predictions = np.loadtxt('../../results/cross_valid_predictions_gpr.txt', delimiter=',');
predictions = np.loadtxt(base+target+'.txt', delimiter=',');
corr = np.corrcoef(predictions, ratings)[0, 1]
print corr
