#python3

import sys
import numpy as np

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

data_name= str(sys.argv[-1])

MICE_GLM_MSEs = []
MICE_CART_MSEs = []
MICE_CARTX_MSEs = []
for _ in range(10): 

	trainM = np.loadtxt("{}/train_missing_{}.csv".format(data_name,_), delimiter=",")
	testM = np.loadtxt("{}/test_missing_{}.csv".format(data_name,_), delimiter=",")

	trainX = np.loadtxt("{}/train_data_{}.csv".format(data_name,_), delimiter=",")
	testX = np.loadtxt("{}/test_data_{}.csv".format(data_name,_), delimiter=",")

	# Make incomplete datasets

	trainMnan =np.copy(trainM)
	testMnan = np.copy(testM)
	trainMnan[trainMnan == 0] = np.nan
	testMnan[testMnan == 0] = np.nan

	trainXM = trainX*trainMnan
	testXM = testX*testMnan

	# MICE-GLM

	mice_glm = IterativeImputer(max_iter=10, random_state=0, tol=0.01, estimator=LogisticRegression()) 
	mice_glm.fit(trainXM)

	MSE_test_loss = np.mean(((1-testM) * testX - (1-testM)*mice_glm.transform(testXM))**2) / np.mean(1-testM)
	print('Test RMSE: {:.4}'.format(np.sqrt(MSE_test_loss)))

	MICE_GLM_MSEs.append(MSE_test_loss)

	# MICE-CART

	mice_cart = IterativeImputer(max_iter=10, random_state=0, tol=0.01, estimator=DecisionTreeClassifier(random_state=0)) 
	mice_cart.fit(trainXM)

	MSE_test_loss = np.mean(((1-testM) * testX - (1-testM)*mice_cart.transform(testXM))**2) / np.mean(1-testM)
	print('Test RMSE: {:.4}'.format(np.sqrt(MSE_test_loss)))

	MICE_CART_MSEs.append(MSE_test_loss)

	# MICE-CARTX

	mice_cartx = IterativeImputer(max_iter=10, random_state=0, estimator=ExtraTreesClassifier(n_estimators=10,random_state=0)) 
	mice_cartx.fit(trainXM)

	MSE_test_loss = np.mean(((1-testM) * testX - (1-testM)*mice_cartx.transform(testXM))**2) / np.mean(1-testM)
	print('Test RMSE: {:.4}'.format(np.sqrt(MSE_test_loss)))

	MICE_CARTX_MSEs.append(MSE_test_loss)

print('MICE-GLM Mean Test RMSE: {:.4}'.format(np.mean(np.sqrt(MICE_GLM_MSEs))))
print('MICE-GLM SD: {:.4}'.format(np.std(np.sqrt(MICE_GLM_MSEs))))

print('MICE-CART Mean Test RMSE: {:.4}'.format(np.mean(np.sqrt(MICE_CART_MSEs))))
print('MICE-CART SD: {:.4}'.format(np.std(np.sqrt(MICE_CART_MSEs))))

print('MICE-CARTX Mean Test RMSE: {:.4}'.format(np.mean(np.sqrt(MICE_CARTX_MSEs))))
print('MICE-CARTX SD: {:.4}'.format(np.std(np.sqrt(MICE_CARTX_MSEs))))