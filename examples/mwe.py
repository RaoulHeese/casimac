from casimac.casimac import CASIMAClassifier, __version__
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

# Create toy data
N = 10
seed = 42
X = np.random.RandomState(seed).uniform(-10,10,N).reshape(-1,1)
y = np.zeros(X.size)
y[X[:,0]>0] = 1

# Classify
clf = CASIMAClassifier(GaussianProcessRegressor)
clf.fit(X, y)
acc = clf.score(X,y)

# Predict
y_predict = clf.predict(X)
p_predict = clf.predict_proba(X)

# Output
print(f'version:      {__version__}')
print(f'accuracy:     {acc}')
print(f'ground truth: {y}')
print(f'prediction:   {y_predict}')
print(f'p(y=1):       {p_predict[:,1]}')