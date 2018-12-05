import numpy as np

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process
'''
#calculating pearson correlation
ratings = np.loadtxt('../../data/ratings.txt', delimiter=',')
predictions = np.loadtxt('../../results/cross_valid_predictions_gpr.txt', delimiter=',');
corr = np.corrcoef(predictions, ratings)[0, 1]
print corr
'''
## read data

features = np.loadtxt('../../data_one_side/features_ALL.txt', delimiter=',')
ratings = np.loadtxt('../../data/ratings.txt', delimiter=',')
predictions = np.zeros(ratings.size);

for i in range(0, 500):
	features_train = np.delete(features, i, 0)
	features_test = features[i, :]
	ratings_train = np.delete(ratings, i, 0)
	ratings_test = ratings[i]
	pca = decomposition.PCA(n_components=100)
	pca.fit(features_train)
	features_train = pca.transform(features_train)
	features_test = pca.transform(features_test.reshape(1,-1))
	regr = linear_model.BayesianRidge()
	regr.fit(features_train, ratings_train)
	predictions[i] = regr.predict(features_test.reshape(1,-1))
	print(predictions[i])
	print 'number of models trained:', i+1

np.savetxt('../../results_one_side/cross_valid_predictions_Raw_Lasso.txt', predictions, delimiter=',', fmt = '%.04f')
corr = np.corrcoef(predictions, ratings)[0, 1]
print corr
