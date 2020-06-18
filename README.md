# CASIMAC Classifier

CASIMAC stands for *Calibrated Simplex Mapping Classifier*.

It is a multi-class/single-label classification algorithm, which allows the prediction of class labels and their probabilities. The classifier is designed along the principles of an [scitkit-learn](https://scikit-learn.org) estimator.

## Getting Started

Use the ``CASIMAClassifier`` class to create a classifier object. This object provides a ``fit`` method for training and a ``predict`` method for the estimation of class labels. Furthermore, the ``predict_proba`` method can be used to predict class label probabilities.

Below is a short example.

```
from casimac import CASIMAClassifier

# Third party packages
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

# Create data
N = 10
seed = 42
X = np.random.RandomState(seed).uniform(-10,10,N).reshape(-1,1)
y = np.zeros(X.size)
y[X[:,0]>0] = 1

# Classify
clf = CASIMAClassifier(GaussianProcessRegressor)
clf.fit(X, y)

# Predict
X_sample = np.linspace(-10,10,100).reshape(-1,1)
y_sample = clf.predict(X_sample)
p_sample = clf.predict_proba(X_sample)
                       
# Plot result
plt.figure(figsize=(10,5))
plt.plot(X_sample,y_sample,label="class prediction")
plt.plot(X_sample,p_sample[:,1],label="class probability prediction")
plt.scatter(X,y,c='r',label="train data")
plt.xlabel("X")
plt.ylabel("label / probability")
plt.legend()
plt.show()
```

Also implemented are a ``fit_transform`` method and an ``inverse_transform`` method to map the latent variables to a unit simplex and vice versa. These methods work only on an already fitted classifier object.

### Prerequisites

The implementation is designed for Python 3. It depends on [scikit-learn](https://scikit-learn.org/stable/install.html) of version 0.21.2 or higher and [SciPy](https://www.scipy.org/install.html) of version 1.2.3 or higher. Also required are other standard packages.

### Installing

Clone the repository and run the example code above to verify that it works.

## License

This project is licensed under the MIT License - see the [LICENSE file](LICENSE) for details.