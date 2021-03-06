import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='linear_model')
parser.add_argument('-featuredim', type=int, default=100)
parser.add_argument('-inputfeatures', type=str, default='../../data/2500features_ALL.txt')
parser.add_argument('-labels', type=str, default='../../data/2500ratings.txt')
args = parser.parse_args()


features = np.loadtxt(args.inputfeatures, delimiter=',')
#features = preprocessing.scale(features)
print(features.shape)
features_train = features[0:-500]
features_test = features[-500:]

#test = features[-1]

pca = decomposition.PCA(n_components=args.featuredim)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)
#test = pca.transform(test.reshape(1,-1))

ratings = np.loadtxt(args.labels, delimiter=',')
print(ratings.shape)
#ratings = preprocessing.scale(ratings)
ratings_train = ratings[0:-500]
ratings_test = ratings[-500:]

if args.model == 'linear_model':
	regr = linear_model.Lasso()

elif args.model == 'svm':
	regr = svm.SVR()
elif args.model == 'rf':
	regr = RandomForestRegressor(n_estimators=50, random_state=0)
elif args.model == 'gpr':
	regr = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
else:
	raise NameError('Unknown machine learning model. Please us one of: rf, svm, linear_model, gpr')

regr.fit(features_train, ratings_train)
ratings_predict = regr.predict(features_test)
corr = np.corrcoef(ratings_predict, ratings_test)[0, 1]
print 'Correlation:', corr

residue = np.mean((ratings_predict - ratings_test) ** 2)
print 'Residue:', residue

#print 'Test:' , regr.predict(test) 

truth, = plt.plot(ratings_test, 'r')
prediction, = plt.plot(ratings_predict, 'b')
plt.legend([truth, prediction], ["Ground Truth", "Prediction"])

plt.show()
'''
(2500L, 11628L)
(2500L,)
Correlation: 0.4575345476255592
Residue: 0.40924107679494237
'''