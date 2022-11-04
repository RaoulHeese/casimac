**********************************************
CASIMAC: Calibrated Simplex Mapping Classifier
**********************************************

.. image:: https://readthedocs.org/projects/casimac/badge/?version=latest
    :target: https://casimac.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
	
.. image:: https://img.shields.io/pypi/v/casimac
    :target: https://pypi.org/project/casimac/
    :alt: PyPI - Project
	
.. image:: https://img.shields.io/badge/license-MIT-lightgrey
    :target: https://github.com/RaoulHeese/casimac/blob/main/LICENSE
    :alt: MIT License	
	
.. image:: https://raw.githubusercontent.com/RaoulHeese/casimac/master/docs/source/_static/simplex.png
    :align: center
	
This Python project provides a supervised multi-class/single-label classification algorithm, which allows the prediction of class labels and their probabilities including gradients with respect to features. The classifier is designed along the principles of an `scikit-learn <https://scikit-learn.org>`_ estimator.

The details of the algorithm have been published in `arXiv:2103.02926 <https://arxiv.org/abs/2103.02926>`_.

Complete documentation of the code is available via `<https://casimac.readthedocs.io/en/latest/>`_.

**Installation**

Install the package via pip or clone this repository. In order to use pip, type:

.. code-block:: sh

  $ pip install casimac

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

Also implemented are a ``fit_transform`` method and an ``inverse_transform`` method to map the latent variables to a unit simplex and vice versa. These methods work only on an already fitted classifier object. Example notebooks can be found in the `examples` directory.

📖 **Citation**

If you find this code useful, please consider citing:

.. code-block::

  @misc{heese2021calibrated,
        title={Calibrated Simplex Mapping Classification}, 
        author={Raoul Heese and Micha{\l} Walczak and Michael Bortz and Jochen Schmid},
        year={2021},
        eprint={2103.02926},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
       }

**License**

This project is licensed under the MIT License - see the LICENSE file for details.