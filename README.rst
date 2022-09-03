**********************************************
CASIMAC: Calibrated Simplex Mapping Classifier
**********************************************

This Python project provides a supervised multi-class/single-label classification algorithm, which allows the prediction of class labels and their probabilities. The classifier is designed along the principles of an https://scikit-learn.org estimator.

**Getting Started**

Use the ``CASIMAClassifier`` class to create a classifier object. This object provides a ``fit`` method for training and a ``predict`` method for the estimation of class labels. Furthermore, the ``predict_proba`` method can be used to predict class label probabilities.

Below is a short example.

.. code-block:: python

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

Also implemented are a ``fit_transform`` method and an ``inverse_transform`` method to map the latent variables to a unit simplex and vice versa. These methods work only on an already fitted classifier object.

**Prerequisites**

The implementation is designed for Python 3. It depends on (https://scikit-learn.org/stable/install.html) of version 0.21.2 or higher and (https://www.scipy.org/install.html) of version 1.2.3 or higher. Also required are other standard packages.

**Installation**

Clone the repository and run the example code above to verify that it works.

**Reference**

The algorithm is an implementation from our paper "Calibrated Simplex Mapping Classification". Preprint available via https://arxiv.org/abs/2103.02926. If you find this code useful, please consider citing:

.. code-block::

  @misc{heese2021calibrated,
		title={Calibrated Simplex Mapping Classification}, 
		author={Raoul Heese and Michał Walczak and Michael Bortz and Jochen Schmid},
		year={2021},
		eprint={2103.02926},
		archivePrefix={arXiv},
		primaryClass={stat.ML}
  }

**License**

This project is licensed under the MIT License - see the LICENSE file) for details.