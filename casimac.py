# -*- coding: utf-8 -*-
"""CASIMAC Classifier. 

Author: Raoul Heese 

Created on Thu Jun 1 12:00:00 2020
"""


__version__ = "1.0.0"


import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import NotFittedError
from scipy.special import erf


class CASIMAClassifier(BaseEstimator, ClassifierMixin):   
    """Multi-class/single-label classifier.
    
    Parameters
    ----------
    
    model_constructor : callable
        Method that returns an sklearn estimator. This estimator is trained on 
        the estimation of latent variables from features. In particular, the 
        estimator must provide a ``fit`` method (for training) and a 
        ``predict`` method (for predictions). For the prediction of class 
        probabilities, the ``predict`` method must also support a second 
        argument ``return_std``, which returns the standard deviations of the 
        predictions together with the mean values if set to True. It is 
        assumed that the predictions of the estimator obey a Gaussian 
        probability distribution with the aforementioned mean and variance.
        
    repulsion_strength : float, optional (default: 1)
        Scalar strength used for the repulsion term. Should be non-negative. 
        Choose 0 to disable repulsions.
        
    repulsion_number : int, optional (default: 1)
        Number of nearest neighbors used for the repulsion term.   
        
    repulsion_reduce : callable, optional  (default: np.nanmean)
        Function to reduce the set of nearest neighbor distances to a single 
        number used in the repulsion term. Note that np.nan may occur in the 
        list of distances.
        
    repulsion_fun : callable or None, optional (default: None)
        Final function that is applied to the repulsion term.  Set to None to 
        disable the function call.
        
    attraction_strength : float, optional (default: 1)
        Scalar strength used for the attraction term. Should be non-negative. 
        Choose 0 to disable attraction.
        
    attraction_number : int, optional (default: 1)
        Number of nearest neighbors used for the attraction term.    
        
    attraction_reduce : callable, optional  (default: np.nanmean)
        Function to reduce the set of nearest neighbor distances to a single 
        number used in the attraction term. Note that np.nan may occur in the 
        list of distances.        
        
    attraction_fun : callable or None, optional (default: np.reciprocal)
        Final function that is applied to the attraction term. Set to None to 
        disable the function call.    
        
    c_transformation_fun : callable or None, optional (default: None)
        Optional transformtion function (e.g., for rescaling) of the latent 
        variable coefficients. Set to None to disable the function call.        
    
    metric : str or callable, optional (default: 'euclidean')
        Metric options used in ``sklearn.metrics.pairwise_distances``. See the 
        respective documentation for more details.
    
    proba_calc_method : 'analytical', 'MC' or 'MC-fast', optional (default: 
        'analytical')
        Determines the method used for the prediction of class probabilities. 
        Choose 'analytical' for an analytical calculation (can only be used 
        for two classes, otherwise fall back to 'MC'). Choose 'MC' for a 
        sequential Monte Carlo implementation (slower, less memory) and 
        'MC-fast' for a simultaneous Monte Carlo implementation (faster, 
        more memory).

    proba_NMC : int, optional (default: 1000)
        Number of Monte Carlo samples (per dimension) for the prediciton of 
        class probabilities.
        
    random_state : int, RandomState instance or None, optional (default: None)
        The random generator to use for the prediction of class probabilities. 
        If an integer is given, a new random generator with this seed is 
        created.
        
    verbose : bool, optional (default: None)
        Set to true to enable diagnostic messages. Currently not implemented.
        
    Attributes
    ----------
    
    X_ : array-like of shape (n_samples, n_features)
        Feature vectors in training data.
        
    y_ : array-like of shape (n_samples,)
        Target labels in training data.
        
    classes_ : array of shape (n_classes,)
        Unique class labels in y_.
    
    d_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Vector of latent variables calculated from X_ and y_.
        
    model_ : obj
        Instance of the model trained on the estimation of latent variables 
        from features. Is created by the call of model_constructor.
    
    random_state_ : np.random.RandomState
        Instance of the random state used for Monte Carlo predictions of 
        the class probabilities.
    """
    
    def __init__(self, model_constructor, repulsion_strength=1, repulsion_number=1, 
                 repulsion_reduce=np.nanmean, repulsion_fun=None, attraction_strength=0,
                 attraction_number=1, attraction_reduce=np.nanmean, attraction_fun=np.reciprocal,
                 c_transformation_fun=None, metric='euclidean', proba_calc_method='analytical',
                 proba_NMC=1000, random_state=None, verbose=False):
        self.model_constructor = model_constructor
        self.repulsion_strength = repulsion_strength
        self.repulsion_number = repulsion_number
        self.repulsion_reduce = repulsion_reduce
        self.repulsion_fun = repulsion_fun
        self.attraction_strength = attraction_strength
        self.attraction_number = attraction_number
        self.attraction_reduce = attraction_reduce
        self.attraction_fun = attraction_fun
        self.c_transformation_fun = c_transformation_fun
        self.metric = metric
        self.proba_calc_method = proba_calc_method # [MC, MC-fast, analytical]
        self.proba_NMC = proba_NMC # per dimension
        self.random_state = random_state
        self.verbose = verbose
    
    def _calc_class_normals(self, n):
        """Calculate vertices of an (n-1)-simplex, which are stored in the 
        property ``_class_normals``.
        """
        
        # Calculate class normals
        v = np.zeros((n-1,n))
        angle = -1/(n-1)
        v[0,0] = 1
        v[0,1:] = angle
        for i in range(1,n-1):
            v[i,i] = np.sqrt(1 - np.sum(v[:i,i]**2))
            for j in range(i+1,n):
                v[i,j] = (angle - np.dot(v[:i,i],v[:i,j])) / v[i,i]
        
        # Return class normals in rows: (vector 1, ..., vector n), each of dimension n-1
        return v.T
    
    def _calc_latent_coefficients(self, distance_to_own, distance_to_other_list):
        """Calculate coefficients (combined from repulsion and attraction 
        terms) for the transformation to the latent space. Specifically, map 
        from the distance arrays of the own and the other class to an array of 
        reduced distances. This mapping is performed for each class.
        
        Notes:
            1) Nearest neighbor calculation may fail if there are not enough 
               neighbors available.
            2) Returned coefficients must be non-negative.
        """
        
        # Setup coefficients
        c = np.zeros((len(distance_to_other_list), distance_to_own.shape[0]))

        # Repulsion (based on distances to other)
        if self.repulsion_number > 0 and self.repulsion_strength != 0:
            #for idx in range(len(distance_to_other_list)):
            #    distance_to_other_list[idx][distance_to_other_list[idx]<=0] = np.nan # remove invalids
            idx_list = [np.argpartition(d, min(self.repulsion_number, d.shape[1]-1), axis=1) for d in distance_to_other_list]
            repulse = np.array([[self.repulsion_reduce(d[i[:self.repulsion_number]]) for i, d in zip(idx,distances)] for idx, distances in zip(idx_list,distance_to_other_list) ])
            if self.repulsion_fun is not None:
                repulse = self.repulsion_fun(repulse)
            c += self.repulsion_strength * repulse

        # Attraction (based on distance to own)
        if self.attraction_number > 0 and self.attraction_strength != 0:
            distance_to_own[distance_to_own<=0] = np.nan # remove invalids
            idx_list = np.argpartition(distance_to_own, min(self.attraction_number, distance_to_own.shape[1]-1), axis=1)
            attract = np.array([self.attraction_reduce(d[i[:self.attraction_number]]) for i, d in zip(idx_list,distance_to_own)])
            if self.attraction_fun is not None:
                attract = self.attraction_fun(attract)
            attract = np.tile(attract,(c.shape[0],1))
            c += self.attraction_strength * attract
            
        # Return coefficients
        if self.c_transformation_fun is not None:
            c = self.c_transformation_fun(c)
        return c
    
    def _calc_distance_features(self, X, y):
        """Calculate distance features.
        """
        
        d = np.zeros((X.shape[0],self._num_classes-1))
        D = pairwise_distances(X, metric=self.metric)
        X_class_idxa_list = [np.where(y == self.classes_[class_idx])[0] for class_idx in range(self._num_classes)]
        for class_idx in range(self._num_classes):
            other_class_idx_list = [other_class_idx for other_class_idx in range(self._num_classes) if other_class_idx != class_idx]
            distance_to_own = D[np.array(X_class_idxa_list[class_idx]).reshape(-1,1),X_class_idxa_list[class_idx]]
            distance_to_other_list = [D[np.array(X_class_idxa_list[class_idx]).reshape(-1,1),np.array(X_class_idxa_list[other_class_idx])] for other_class_idx in other_class_idx_list]
            c = self._calc_latent_coefficients(distance_to_own, distance_to_other_list)
            d[X_class_idxa_list[class_idx]] = -np.tensordot(c.T,self._class_normals[other_class_idx_list,:],axes=1) # transformation
        return d
    
    def _calc_distance_features_to_class(self, d):
        """Map from distance feature space d to class space y.
        Minimize distance to edge points to determine the correct classes.
        """
        
        d = np.asarray(d).reshape(-1,self._num_classes-1)
        edge_distances = np.zeros((d.shape[0],self._num_classes))
        for j in range(self._num_classes):
            edge_distances[:,j] = np.linalg.norm(self._class_normals[j,:]-d, axis=1)
        best_classes = np.array(np.argmin(edge_distances,axis=1),dtype=np.int64)
        return np.array(self.classes_)[best_classes]
        
    def _calc_proba_mc(self, mu, sigma, return_std, method):
        """Calculate (binary or multi-class) class probabilities (and their 
        standard devitions) with a Monte Carlo approach.
        """

        # Setup
        p = np.zeros((mu.shape[0],self._num_classes)) # p[X index, class index]
        N = self.proba_NMC
        
        # Check shape of sigma: multivariate or univariate (i.e., a uniform gaussian), reshape accordingly
        if sigma.size != mu.size:
            sigma = np.repeat(sigma,mu.shape[1])
           
        # Run MC: Both methods should lead to the same result.
        if method == 'simultaneous':
            # Method 1: Simultaneous calculation of query points. Fast, but needs more memory. (Try to store all in matrix v to reduce memory.)
            v = mu.ravel() # v = mu
            v = self.random_state_.normal(v,sigma.ravel(),(N,mu.size)).reshape((N*mu.shape[0],*mu.shape[1:])) # v = d
            v = self._calc_distance_features_to_class(v).reshape((N,mu.shape[0])) # v = y
            for class_idx in range(self._num_classes):
                p[:,class_idx] = np.sum(v==self.classes_[class_idx],axis=0) / N
                
        elif method == 'sequential':
            # Method 2: Sequential calculation of query points. Slower, but needs less memory.
            sigma = sigma.reshape(mu.shape)
            for idx in range(mu.shape[0]):
                d = self.random_state_.normal(mu[idx,:], sigma[idx,:], (N, mu.shape[1]))
                y = self._calc_distance_features_to_class(d)
                for class_idx in range(self._num_classes):
                    p[idx,class_idx] = y[y==self.classes_[class_idx]].size / N
        else:
            raise NotImplementedError("_calc_proba_mc method '{}' not implemented!".format(method))
        
        # Return result     
        if return_std:
            p_sigma = np.sqrt((p * (1-p)**2 + (1-p) * p**2) / (N-1))
            return p, p_sigma
        else:
            return p         
    
    def _calc_proba_analytical(self, mu, sigma, return_std):
        """Calculate binary class probabilities (and their standard devitions)
        with analytical formulas.
        """
        mu = mu.ravel()
        sigma = sigma.ravel()
        sigma[sigma==0] = np.nan # handle sigma = 0
        p_pos = (1 + erf(np.abs(mu.ravel())/(np.sqrt(2)*sigma)))/2
        p_pos[np.isnan(p_pos)] = 1 # handle sigma = 0
        p_pos = p_pos.reshape(-1,1)
        p_neg = np.zeros(p_pos.shape)
        p_neg[mu < 0] = p_pos[mu < 0]
        p_pos[mu < 0] = 1 - p_neg[mu < 0]
        p_neg[mu >= 0] = 1 - p_pos[mu >= 0] # =0: bias towards pos class
        p = np.concatenate((p_pos,p_neg), axis = 1)
        if return_std:
            p_sigma = np.zeros(p.shape)
            return p, p_sigma
        else:
            return p
            
    def _calc_proba(self, mu, sigma, return_std):
        """Call suitable probability calculator depending on options.
        """
        
        if self.proba_calc_method == 'analytical':
             if self._num_classes == 2:
                 self.used_proba_calc_method_ = 'analytical' 
                 return self._calc_proba_analytical(mu, sigma, return_std)
             else:
                 warnings.warn("Probability calculation method 'analytical' can only be used for 2 classes; here we have {} classes. Fall back to calculation method 'MC'.".format(self._num_classes), stacklevel=2)
                 self.used_proba_calc_method_ = 'MC'
                 return self._calc_proba_mc(mu, sigma, return_std, "sequential")
        elif self.proba_calc_method == 'MC':
            self.used_proba_calc_method_ = 'MC'
            return self._calc_proba_mc(mu, sigma, return_std, "sequential")
        elif self.proba_calc_method == 'MC-fast':
            self.used_proba_calc_method_ = 'MC-fast'
            return self._calc_proba_mc(mu, sigma, return_std, "simultaneous")
        else:
            raise NotImplementedError("Unknown probability calculation method '{}'!".format(self.proba_calc_method))
            
    def _calc_default_tau(self, d):
        """Calculate the data-dependent scaling for transformations.
        """
        
        return 1/np.min(np.std(d,axis=0))            

    def fit(self, X, y, d=None):
        """Fit Classifier.
         
        Note that the estimator may depend on the naming of the labels. That 
        is, because the set of unique labels (stored in the attribute 
        ``classes_``) determines the association of classes to simplex 
        vertices and therefore different associations lead to different latent 
        spaces. All these latent spaces are linearly homeomorphic to each 
        other, but can lead to a different behavior of the regression model 
        (stored in the attribute ``model_``).
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Feature vectors of training data.
            
        y : array-like of shape (n_samples,)
            Target labels of training data.
            
        d : latent variables, array-like of shape (n_samples, n_classes-1) 
            or None, optional (default: None)
            Precalculated vector of latent variables. Set to None to calculate 
            d automatically based on X and y.
            
        Returns
        -------
        
        self : returns an instance of self.
        """
       
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(self.y_)
        self._num_classes = self.classes_.size
        if self._num_classes < 2:
            raise ValueError("At least 2 classes required!")
        self._class_normals = self._calc_class_normals(self._num_classes)
        
        # Prepare rng
        self.random_state_ = check_random_state(self.random_state)
        
        # Prepare model
        self.model_ = self.model_constructor()
        if not hasattr(self.model_, 'fit') or not hasattr(self.model_, 'predict'):
            raise ValueError("Model must provide methods 'fit' and 'predict'!")
        
        # Distance features
        self.d_ = d
        if self.d_ is None:
            self.d_ = self._calc_distance_features(self.X_,self.y_)
        else:
            self.d_ = np.asarray(self.d_)
            if self.d_.shape != (self.X_.shape[0],self._num_classes-1):
                raise ValueError("Provided distance features have an invalid shape: {} instead of {}!".format(self.d_.shape, (self.X_.shape[0],self._num_classes-1)))
        
        # Fit model
        self.model_.fit(self.X_,self.d_)
        
        # Return the classifier
        return self
    
    def predict(self, X):
        """Perform classification on an array of test vectors X. Requires a 
        previous call of ``fit``.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Query points where the classifier is evaluated.
            
        Returns
        -------
        
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``.
        """
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])
        
        # Make model prediction
        d_predict = self.model_.predict(X)
        return self._calc_distance_features_to_class(d_predict)
    
    def predict_proba(self, X, return_std=False):
        """Return probability estimates for the test vector X. Requires a 
        previous call of ``fit``.
        
        Note that it is assumed that the predictions of the regression model 
        (stored in the attribute ``model_``) obey a Gaussian probability 
        distribution. The ``predict`` method of the regression model must 
        support a second argument ``return_std``, which returns the standard 
        deviations of the predictions together with the mean values if set to 
        True so that ``(mean, std) = model_.predict(X, return_std=True)``.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Query points where the classifier is evaluated.
            
        return_std : bool, optional (default: False)
            If True, the standard-deviation of the predictive distribution at 
            the query points is returned along with the mean.
            
        Returns
        -------
        
        p : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
            
        p_std : array-like of shape (n_samples,), optional
            Best estimate of the standard deviation of the predicted 
            probabilities at the query points.
            Only returned when return_std is True.            
        """

        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])
        
        # Make model prediction
        mu, sigma = self.model_.predict(X, return_std=True)
        return self._calc_proba(mu, sigma, return_std)

    def fit_transform(self, tau=None):
        """Transforms all latent space coordinates to the probability simplex 
        space (dimensions+1). This corresponds to ``transform(self.d_, tau)``. 
        Also store the scaling factor in the attribute ``tau_``. Requires a 
        previous call of ``fit``.
        
        Parameters
        ----------
            
        tau : float or None, optional (default: None)
            Scaling factor. If set to None, a data-dependent scaling is used.
            
        Returns
        -------
        
        s : array-like of shape (n_samples, n_classes+1)
            Simplex vector space coordinates as a representation of the 
            attribute ``d_``.
        """

        # Check if fitted
        check_is_fitted(self, ['d_', 'y_']) 
        
        # Perform transformation and store results
        unit_simplex, tau = self.transform(self.d_, tau, return_tau=True)
        self.tau_ = tau
        return unit_simplex
    
    def transform(self, d, tau=None, return_tau=False):
        """Transform latent space coordinates to the probability simplex space 
        (dimensions+1). Requires a previous call of ``fit``.
        
        Parameters
        ----------

        d : array-like of shape (n_samples, n_classes)
            Latent space coordinates to transform.
        
        tau : float or None, optional (default: None)
            Scaling factor. If set to None, a data-dependent scaling is used.
            
        return_tau : bool, optional (default: False)
            If set to True, return the used scaling factor.
            
        Returns
        -------
        
        s : array-like of shape (n_samples, n_classes+1)
            Simplex vector space coordinates as a representation of the 
            attribute ``d_``.
            
        tau : float
            Scaling factor used for the transformation.
            Only returned when return_tau is True.  
        """        
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_']) 
        
        # Perform transformation
        if tau is None: # use default data-dependent scaling
            tau = self._calc_default_tau(d)
        projections = np.zeros((d.shape[0],self._num_classes))
        for i in range(self._num_classes):
            projections[:,i] = np.dot(d*tau,self._class_normals[i,:])
        transformed_projections = np.exp(projections)
        unit_simplex_norm = np.repeat(np.sum(transformed_projections,axis=1),self._num_classes).reshape((d.shape[0],self._num_classes))
        unit_simplex = transformed_projections / unit_simplex_norm
        if return_tau:
            return unit_simplex, tau
        return unit_simplex

    def inverse_transform(self, s, tau=None):
        """Transform back from the probability simplex space to the latent 
        space. Requires a previous call of ``fit``.
        
        Parameters
        ----------
           
        s : array-like of shape (n_samples, n_classes+1)
            Simplex vector space coordinates to transform.
            
        tau : float or None, optional (default: None)
            Scaling factor. If set to None, the previously fitted scaling from 
            ``fit_transform`` is used, which is stored in the attribute 
            ``tau_``.         
            
        Returns
        -------
        
        d : array-like of shape (n_samples, n_classes)
            Inverse transformation of the simplex vector space coordinates s.
        """
        
        check_is_fitted(self, ['d_', 'y_'])   
        if tau is None: # use previously fitted scaling
            if not hasattr(self, 'tau_'):
                raise NotFittedError("The transformation is not fitted yet. Call 'fit_transform' with appropriate arguments or provide a fixed value for tau.")
            tau = self.tau_
        d = np.zeros((s.shape[0],self._num_classes-1))
        projections = np.log(s) # shifted by np.sum(shifted_projections, axis=1)/self._num_classes (however, this shift is eliminated by summing over the outer product)
        projection_normal = (self._num_classes-1)/self._num_classes
        for i in range(self._num_classes):
            d += np.outer(projections[:,i],self._class_normals[i,:])
        d *= projection_normal / tau
        return d