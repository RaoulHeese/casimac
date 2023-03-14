# -*- coding: utf-8 -*-
"""CASIMAC: multi-class/single-label classifier with gradients. 
"""


__version__ = "1.2.3"


import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import pairwise_distances
from scipy.special import erf
from scipy import optimize


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
        Scalar strength used for the repulsion term (``beta``). Should be 
        non-negative. Choose 0 to disable repulsions.
        
    repulsion_number : int, optional (default: 1)
        Number of nearest neighbors used for the repulsion term (``k_beta``).    
        
    attraction_strength : float, optional (default: 1)
        Scalar strength used for the attraction term (``alpha``). Should be 
        non-negative. Choose 0 to disable attraction.
        
    attraction_number : int, optional (default: 1)
        Number of nearest neighbors used for the attraction term (``k_alpha``).    
    
    metric : str or callable, optional (default: 'euclidean')
        Metric options used in ``sklearn.metrics.pairwise_distances``. See the 
        respective documentation for more details.
    
    proba_calc_method : 'analytical', 'MC' or 'MC-fast', optional (default: 'analytical')
        Determines the method used for the prediction of class probabilities. 
        Choose 'analytical' for an analytical calculation (can only be used 
        for two classes, otherwise fall back to 'MC'). Choose 'MC' for a 
        sequential Monte Carlo implementation (slower, less memory) and 
        'MC-fast' for a simultaneous Monte Carlo implementation (faster, 
        more memory).

    proba_NMC : int, optional (default: 1000)
        Number of Monte Carlo samples (per dimension) for the prediciton of 
        class probabilities.
        
    p_calc_method: 'iterative', 'explicit', optional (default: 'iterative')
        Determines the method for the calculation of the simplex vectors.
        
    random_state : int, RandomState instance or None, optional (default: None)
        The random generator to use for the prediction of class probabilities. 
        If an integer is given, a new random generator with this seed is 
        created. ``None`` leads to a newly generated seed.
        
    l_repulsion_reduce : callable, optional (default: numpy.nanmean)
        Legacy option, not recommended! Function to reduce the set of nearest 
        neighbor distances to a single number used in the repulsion term. Note 
        that numpy.nan may occur in the list of distances.
        
    l_repulsion_fun : callable or None, optional (default: None)
        Legacy option, not recommended! Final function that is applied to the 
        repulsion term. Set to ``None`` to disable the function call. 
        
    l_attraction_reduce : callable, optional (default: numpy.nanmean)
        Legacy option, not recommended! Function to reduce the set of nearest 
        neighbor distances to a single number used in the attraction term. Note 
        that numpy.nan may occur in the list of distances.        
        
    l_attraction_fun : callable or None, optional (default: numpy.reciprocal)
        Legacy option, not recommended! Final function that is applied to the 
        attraction term. Set to ``None`` to disable the function call.    
        
    l_c_transformation_fun : callable or None, optional (default: None)
        Legacy option, not recommended! Optional transformtion function (e.g., 
        for rescaling) of the latent variable coefficients. Set to ``None`` to 
        disable the function call.   
        
    Attributes
    ----------
    
    X_ : array-like of shape (n_samples, n_features)
        Feature vectors in training data.
        
    y_ : array-like of shape (n_samples,)
        Target labels in training data.
        
    classes_ : array of shape (n_classes,)
        Unique class labels in y\_.
    
    d_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Vector of latent variables calculated from X\_ and y\_.
        
    model_ : obj
        Instance of the model trained on the estimation of latent variables 
        from features. Is created by the call of model_constructor.
    
    random_state_ : numpy.random.RandomState
        Instance of the random state used for Monte Carlo predictions of 
        the class probabilities.
    """
    
    def __init__(self, model_constructor,
                 repulsion_strength=1, repulsion_number=1, 
                 attraction_strength=0, attraction_number=1, 
                 metric='euclidean', proba_calc_method='analytical',
                 proba_NMC=1000, p_calc_method='iterative', random_state=None,
                 l_repulsion_reduce=np.nanmean, l_repulsion_fun=None,
                 l_attraction_reduce=np.nanmean, l_attraction_fun=np.reciprocal,
                 l_c_transformation_fun=None):
        
        # Settings
        self.model_constructor = model_constructor
        self.repulsion_strength = repulsion_strength
        self.repulsion_number = repulsion_number
        self.attraction_strength = attraction_strength
        self.attraction_number = attraction_number
        self.metric = metric
        self.proba_calc_method = proba_calc_method # [MC, MC-fast, analytical]
        self.proba_NMC = proba_NMC # per dimension
        self.p_calc_method = p_calc_method
        self.random_state = random_state
        self.l_repulsion_reduce = l_repulsion_reduce # legacy option
        self.l_repulsion_fun = l_repulsion_fun # legacy option
        self.l_attraction_reduce = l_attraction_reduce # legacy option
        self.l_attraction_fun = l_attraction_fun # legacy option
        self.l_c_transformation_fun = l_c_transformation_fun # legacy option
            
    def _calc_class_normals(self, n):
        """Calculate class normals (i.e., the negative vertices) of an 
        (n-1)-simplex. The results are stored in the attribute 
        ``_class_normals`` during a call of ``fit``.
        """        
        
        # Choose calculation method
        if self.p_calc_method == 'iterative':
            return self._calc_class_normals_iterative(n)
        elif self.p_calc_method == 'explicit':
            return self._calc_class_normals_explicit(n)
        else:
            raise NotImplementedError("Unknown class normal calculation method '{}'!".format(self.p_calc_method))
    
    def _calc_class_normals_iterative(self, n):
        """Calculate the class normals (i.e., the negative vertices) of an 
        (n-1)-simplex using an iterative method.
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
    
    def _calc_class_normals_explicit(self, n):
        """Calculate the class normals (i.e., the negative vertices) of an 
        (n-1)-simplex using an explicit method.
        """
        
        # Calculate simplex vertices
        q = np.zeros((n,n-1))
        for i in range(1,n):
            q[i-1,i-1] = 1
        q[n-1,:] = (1+np.sqrt(n))/(n-1)
        v = np.empty((n,n-1))
        c = (1+1/np.sqrt(n))/(n-1)
        nu = np.sqrt(1-1/n)
        for i in range(1,n+1):
            v[i-1,:] = (q[i-1] - c) / nu
        
        # Return class normals in rows: (vector 1, ..., vector n), each of dimension n-1
        return -v

    def _calc_binary_projectors(self):
        """Calculate binary projectors (normalized segmentation planes), 
        which are used to obtain the decision function. They are stored in the 
        attribute ``_binary_projectors`` during a call of ``fit``.
        """
        
        # Calculate (normalized) binary projectors
        bp = np.zeros((self._num_classes,self._num_classes,self._num_classes-1))
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i != j:
                    bp[i,j,:] = self._class_normals[i,:]-self._class_normals[j,:]
                    bp[i,j,:] /= np.linalg.norm(bp[i,j,:])
        
        # Return projectors as array [class i, class j, vector]
        return bp
    
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
            repulse = np.array([[self.l_repulsion_reduce(d[i[:self.repulsion_number]]) for i, d in zip(idx,distances)] for idx, distances in zip(idx_list,distance_to_other_list) ])
            if self.l_repulsion_fun is not None:
                repulse = self.l_repulsion_fun(repulse)
            c += self.repulsion_strength * repulse

        # Attraction (based on distance to own)
        if self.attraction_number > 0 and self.attraction_strength != 0:
            distance_to_own[distance_to_own<=0] = np.nan # remove invalids
            idx_list = np.argpartition(distance_to_own, min(self.attraction_number, distance_to_own.shape[1]-1), axis=1)
            attract = np.array([self.l_attraction_reduce(d[i[:self.attraction_number]]) for i, d in zip(idx_list,distance_to_own)])
            if self.l_attraction_fun is not None:
                attract = self.l_attraction_fun(attract)
            attract = np.tile(attract,(c.shape[0],1))
            c += self.attraction_strength * attract
            
        # Return coefficients
        if self.l_c_transformation_fun is not None:
            c = self.l_c_transformation_fun(c)
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
    
    def _calc_edge_distances(self, d):
        """Calculate distances to edge points, which can be used to determine
        the correct classes.
        """
        
        d = np.asarray(d).reshape(-1,self._num_classes-1)
        edge_distances = np.zeros((d.shape[0],self._num_classes))
        for j in range(self._num_classes):
            edge_distances[:,j] = np.linalg.norm(self._class_normals[j,:]-d, axis=1)
        return edge_distances
    
    def _calc_distance_features_to_class(self, d):
        """Map from distance feature space d to class space y.
        Minimize distance to edge points to determine the correct classes.
        """
        
        best_classes = np.array(np.argmin(self._calc_edge_distances(d),axis=1),dtype=np.int64) # return first minimum by default
        return np.array(self.classes_)[best_classes] 

    def _calc_proba_analytical(self, mu, sigma, return_std):
        """Calculate binary class probabilities (and their standard deviations)
        with analytical formulas.
        """

        mu = mu.ravel()
        sigma = sigma.ravel()
        sigma[sigma==0] = np.nan
        p_pos = (1+erf(mu/(np.sqrt(2)*sigma)))/2
        p_pos[np.isnan(p_pos)] = (np.sign(mu[np.isnan(p_pos)])+1)/2
        p_neg = 1 - p_pos
        p = np.concatenate((p_pos[:,np.newaxis], p_neg[:,np.newaxis]), axis=1)
        if return_std:
            p_sigma = np.zeros(p.shape)
            return p, p_sigma
        else:
            return p

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
            
    def _calc_proba_grad_analytical(self, mu, sigma, dmu, dsigma):
        """Calculate binary class probability gradients with analytical 
        formulas.
        """

        mu = mu.ravel()
        sigma = sigma.ravel()
        sigma[sigma==0] = np.nan
        dp_pos_mu = np.exp(-mu**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma) # dp/dmu
        dp_pos_mu[np.isnan(dp_pos_mu)] = 0 # ~ np.inf
        dp_pos_sigma = -(mu*np.exp(-mu**2/(2*sigma**2)))/(np.sqrt(2*np.pi)*sigma**2) # dp/dsigma
        dp_pos_sigma[np.isnan(dp_pos_sigma)] = 0 # ~ np.inf
        dp_pos = dp_pos_mu[:,np.newaxis]*dmu[:,:,0] + dp_pos_sigma[:,np.newaxis]*dsigma[:,:,0] # dp
        dp_neg = -dp_pos
        return np.concatenate((dp_pos[:,:,np.newaxis],dp_neg[:,:,np.newaxis]), axis=2)
    
    def _calc_proba_grad_mc(self, mu, sigma, dmu, dsigma):
        """Calculate (binary or multi-class) class probability gradients with a 
        Monte Carlo approach.
        
        Notes:
            1) This method is very experimental and not guaranteed to work.
            2) A more stable method should be used instead.
        """
        
        # TODO: replace with more stable method
        
        # Show warning message
        warnings.warn("The function _calc_proba_grad_mc is very experimental, use with care!")
        
        # Perform gradient calculations
        def mu_func(x, sample_idx, class_idx):
            return self._calc_proba_mc(x, sigma[sample_idx,:].reshape(1,-1))[0,class_idx]
        def sigma_func(x, sample_idx, class_idx):
            return self._calc_proba_mc(mu[sample_idx,:].reshape(1,-1), x)[0,class_idx]
        dp_mu = np.empty((mu.shape[0], self._num_classes, mu.shape[1])) # dp/dmu
        for sample_idx in range(mu.shape[0]):
            for class_idx in range(self._num_classes):
                grad = optimize.approx_fprime(mu[sample_idx,:].reshape(1,-1), mu_func, self._dproba_eps, sample_idx, class_idx)
                dp_mu[sample_idx, class_idx,:] = grad
        dp_sigma = np.empty((mu.shape[0], self._num_classes, mu.shape[1])) # dp/dsigma
        for sample_idx in range(mu.shape[0]):
            for class_idx in range(self._num_classes):
                grad = optimize.approx_fprime(sigma[sample_idx,:].reshape(1,-1), sigma_func, self._dproba_eps, sample_idx, class_idx)
                dp_sigma[sample_idx, class_idx,:] = grad
        dp = np.empty((mu.shape[0], dmu.shape[1], self._num_classes)) # dp
        for var_idx in range(dmu.shape[1]):
            for class_idx in range(self._num_classes):
                dp[:,var_idx,class_idx] = np.sum(dp_mu[:,class_idx,:] * dmu[:,var_idx,:],axis=1) + np.sum(dp_sigma[:,class_idx,:] * dsigma[:,var_idx,:],axis=1) 
        return dp
    
    def _calc_proba_grad(self, mu, sigma, dmu, dsigma):
        """Call suitable probability gradient calculator depending on the 
        number of classes.
        """
        
        if self._num_classes == 2:
            return self._calc_proba_grad_analytical(mu, sigma, dmu, dsigma)
        else:
            return self._calc_proba_grad_mc(mu, sigma, dmu, dsigma) 
        
    def _calc_decision_function(self, d_predict, return_idx_col_map):
        """Calculate decision function.
        """
        
        # determine decision borders for all class combinations
        decision = np.zeros((d_predict.shape[0], self._num_classes*(self._num_classes-1)//2))
        idx_col_map = []
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i < j:
                    decision[:,len(idx_col_map)] = np.dot(d_predict, self._binary_projectors[i,j,:])
                    idx_col_map.append((i,j))
                    
        # return results in a suitable format
        if self._num_classes == 2:
            return decision.ravel() # [n_sample]
        else: # [n_sample, n_class * (n_class-1) / 2], border names in idx_col_map
            if return_idx_col_map:
                return decision, idx_col_map 
            return decision
        
    def _calc_decision_function_grad(self, dmean, return_idx_col_map):
        """Calculate gradient of the decision function.
        """
          
        # determine decision border gradients for all class combinations
        decision_grad = np.zeros((dmean.shape[0], dmean.shape[1], self._num_classes*(self._num_classes-1)//2))
        idx_col_map = []
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i < j:
                    decision_grad[:,:,len(idx_col_map)] = np.dot(dmean, self._binary_projectors[i,j,:])
                    idx_col_map.append((i,j))
                    
        # return results in a suitable format
        if self._num_classes == 2:
            return decision_grad.reshape(dmean.shape[0], dmean.shape[1]) # [n_sample, n_var]
        else: # [n_sample, n_var, n_class * (n_class-1) / 2], border names in idx_col_map
            if return_idx_col_map:
                return decision_grad, idx_col_map 
            return decision_grad
        
    def _transform_ref(self, d, tau):
        """Calculate the reference transformation.
        """
        
        projections = np.zeros((d.shape[0],self._num_classes))
        for i in range(self._num_classes):
            projections[:,i] = np.dot(d*tau,self._class_normals[i,:])
        transformed_projections = np.exp(projections)
        unit_simplex_norm = np.repeat(np.sum(transformed_projections,axis=1),self._num_classes).reshape((d.shape[0],self._num_classes))
        s = transformed_projections / unit_simplex_norm
        return s
    
    def _transform_scale(self, d, tau):
        """Calculate the scale transformation.
        """
        
        s = np.zeros((d.shape[0],self._num_classes))
        for j in range(d.shape[0]):
            exp_list = [np.exp(-1*tau*np.dot(self._class_normals[i,:],d[j,:])) for i in range(self._num_classes)]
            s[j,:] = np.array(exp_list) / np.sum(exp_list)
        return s    

    def _inverse_transform_ref(self, s, tau):
        """Calculate the inverse reference transformation.
        """
        
        d = np.zeros((s.shape[0],self._num_classes-1))
        projections = np.log(s) # shifted by np.sum(shifted_projections, axis=1)/self._num_classes (however, this shift is eliminated by summing over the outer product)
        projection_normal = (self._num_classes-1)/self._num_classes
        for i in range(self._num_classes):
            d += np.outer(projections[:,i],self._class_normals[i,:])
        d *= projection_normal / tau
        return d

    def _inverse_transform_scale(self, s, tau):
        """Calculate the inverse scale transformation.
        """
        
        d = np.zeros((s.shape[0], self._num_classes-1))
        for j in range(s.shape[0]):
            for i in range(self._num_classes):
                d[j,:] -= np.log(s[j,i])*self._class_normals[i,:]
        d *= (self._num_classes-1)/(tau*self._num_classes)
        return d   
    
    def _calc_default_tau(self, d):
        """Calculate the data-dependent scaling factor for transformations.
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
            
        d : latent variables, array-like of shape (n_samples, n_classes-1) or None, optional (default: None)
            Precalculated vector of latent variables. Set to ``None`` to 
            calculate ``d`` automatically based on ``X`` and ``y`` 
            (recommended).
            
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
        self._binary_projectors = self._calc_binary_projectors()
        
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
    
    def predict_proba_grad(self, X):
        """Return the gradient of the probability estimates with respect to the 
        features. Requires a previous call of ``fit``.
        
        Note that it is assumed that the predictions of the regression model 
        (stored in the attribute ``model_``) obey a Gaussian probability 
        distribution. The ``predict`` method of the regression model must 
        support a second argument ``return_std``, which returns the standard 
        deviations of the predictions together with the mean values if set to 
        True so that ``(mean, std) = model_.predict(X, return_std=True)``. 
        Furthermore, the model must provide a function ``predict_grad``, which 
        predicts the gradients of the ``(mean, std)`` predictions from the 
        ``predict`` method with respect to the features in the same way so that
        ``(dmean, dstd) = model_.predict_grad(X, return_std=True)``.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Query points where the classifier is evaluated.
            
        Returns
        -------
        
        dp : array-like of shape (n_samples, n_features, n_classes)
            Returns the gradient of the probability of the samples with respect 
            to each feature for each class in the model.
        """
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])
        
        # Check availability of gradients
        if not hasattr(self.model_, 'predict_grad'):
            raise NotImplementedError("Gradients not available: model does not provide a predict_grad function!")
        
        # Calculate gradients
        mu, sigma = self.model_.predict(X, return_std=True)
        dmean, dstd = self.model_.predict_grad(X, return_std=True)
        return self._calc_proba_grad(mu, sigma, dmean, dstd)
    
    def decision_function(self, X, return_idx_col_map=False):
        """Return the binary decision functions for the test vector X. Requires 
        a previous call of ``fit``.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Query points where the classifier is evaluated.
            
        return_idx_col_map : bool, optional (default: False)
            If True, ``idx_col_map`` is returned.
            
        Returns
        -------
        
        d : array-like of shape (n_samples,) for a binary classification or (n_sample, n_class * (n_class-1) / 2) otherwise
            Returns the decision functions in the form of an array of the form 
            (first class index, second class index) sorted according to 
            idx_col_map. In case of a binary classification problem, the 
            returned array is flattened.
            
        idx_col_map : array-like of shape (n_class*(n_class-1)/2,), optional
            List of tuples (first class index, second class index) to identify 
            the contents of d for a multi-class classification. The indices 
            correspond to the classes in sorted order, as they appear in the 
            attribute ``classes_``.
            Only returned when return_idx_col_map is True and there are more 
            than two classes. In case of two classes, idx_col_map would always 
            correspond to ((0,1),) and is therefore not returned.
        """
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])
        
        # Make model prediction
        d_predict = self.model_.predict(X)
        if len(d_predict.shape) == 1: # ensure correct shape of model output
            d_predict = d_predict[:,None]
        
        # Determine decision function (and optionally the idx_col_map)
        return self._calc_decision_function(d_predict, return_idx_col_map)
    
    def decision_function_grad(self, X, return_idx_col_map=False):
        """Return the gradient of the ecision function with respect to the
        features. Requires a previous call of ``fit``.
        
        Note that it is assumed that the regression model (stored in the 
        attribute ``model_``)  must provide a function ``predict_grad``, which 
        predicts the gradients of the predictions with respect to the features.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Query points where the classifier is evaluated.
            
        return_idx_col_map : bool, optional (default: False)
            If ``True``, ``idx_col_map`` is returned.
            
        Returns
        -------
        
        dd : array-like of shape (n_samples, n_fetaures) for a binary classification or (n_sample, n_features, n_class * (n_class-1) / 2) otherwise.
             Returns the gradient of the decision function with repect to the 
             features.
            
        idx_col_map : array-like of shape (n_class*(n_class-1)/2,), optional
            List of tuples (first class index, second class index) to identify 
            the contents of d for a multi-class classification.
            Only returned when return_idx_col_map is True and there are more 
            than two classes.        
        """
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])
        
        # Check availability of gradients
        if not hasattr(self.model_, 'predict_grad'):
            raise NotImplementedError("Gradients not available: model does not provide a predict_grad function!")
        
        # Calculate gradients
        dmean = self.model_.predict_grad(X)
        return self._calc_decision_function_grad(dmean, return_idx_col_map)

    def fit_transform(self, X, y, d=None, tau=None, method='reference'):
        """Fit the model and transforms all latent space coordinates to another 
        simplex space (dimensions+1). Also store the scaling factor 
        in the attribute ``tau_``. Requires a previous call of ``fit``.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Feature vectors of training data.
            
        y : array-like of shape (n_samples,)
            Target labels of training data.
            
        d : latent variables, array-like of shape (n_samples, n_classes-1) or None, optional (default: None)
            Precalculated vector of latent variables. Set to ``None`` to 
            calculate ``d`` automatically based on ``X`` and ``y`` 
            (recommended).        
            
        tau : float or None, optional (default: None)
            Scaling factor > 0. If set to ``None``, a data-dependent scaling 
            is used (and returned).
            
        method: 'reference' or 'scale', optional (default: 'reference')
            Determines the transformation method. 'reference': transformation 
            of the simplex into rotated cones highlighting the inter-class 
            distances (default method for visualization). 'scale': rescaling 
            of the simplex to a unit simplex.
            
        Returns
        -------
        
        s : array-like of shape (n_samples, n_classes+1)
            Simplex vector space coordinates as a representation of the 
            attribute ``d_``.
            
        tau : float
            Scaling factor used for the transformation.
            Only returned when ``tau`` is set to ``None``.
        """

        # Fit
        self.fit(X, y, d=d)
        
        # Perform transformation
        return self.transform(self.d_, tau, method)
    
    def transform(self, d, tau=None, method='reference'):
        """Transform latent space coordinates to another simplex space 
        (dimensions+1). Requires a previous call of ``fit``.
        
        Parameters
        ----------

        d : array-like of shape (n_samples, n_classes)
            Latent space coordinates to transform.
        
        tau : float or None, optional (default: None)
            Scaling factor > 0. If set to ``None``, a data-dependent scaling 
            is used (and returned).
            
        method: 'reference' or 'scale', optional (default: 'reference')
            Determines the transformation method. 'reference': transformation 
            of the simplex into rotated cones highlighting the inter-class 
            distances (default method for visualization). 'scale': rescaling 
            of the simplex to a unit simplex.

        Returns
        -------
        
        s : array-like of shape (n_samples, n_classes+1)
            Reference simplex vector space coordinates as a representation of 
            the attribute ``d_``.
            
        tau : float
            Scaling factor used for the transformation.
            Only returned when ``tau`` is set to ``None``. 
        """        
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_']) 
        
        # Determine tau
        if tau is None: # use default data-dependent scaling
            tau = self._calc_default_tau(d)
            return_tau = True
        else:
            return_tau = False
        
        # Choose transformation method
        if method == 'reference':
            s = self._transform_ref(d, tau)
        elif method == 'scale':
            s = self._transform_scale(d, tau)
        else:
            raise NotImplementedError("Unknown transformation method '{}'!".format(method))
          
        # Return results
        if return_tau:
            return s, tau
        return s

    def inverse_transform(self, s, tau, method='reference'):
        """Transform back from the transformed simplex space to the latent 
        space. Requires a previous call of ``fit``.
        
        Parameters
        ----------
           
        s : array-like of shape (n_samples, n_classes+1)
            Reference simplex vector space coordinates to transform.
            
        tau : float
            Scaling factor > 0.
            
        method: 'reference' or 'scale', optional (default: 'reference')
            Determines the transformation method. 'reference': transformation 
            of the simplex into rotated cones highlighting the inter-class 
            distances (default method for visualization). 'scale': rescaling 
            of the simplex to a unit simplex.         
            
        Returns
        -------
        
        d : array-like of shape (n_samples, n_classes)
            Inverse transformation of the reference simplex vector space 
            coordinates ``s``.
        """
        
        # Check if fitted
        check_is_fitted(self, ['d_', 'y_'])  
        
        # Choose transformation method
        if method == 'reference':
            return self._inverse_transform_ref(s, tau)
        elif method == 'scale':
            return self._inverse_transform_scale(s, tau)
        else:
            raise NotImplementedError("Unknown transformation method '{}'!".format(method))  
        
    def train(self, X, y, d=None):    
        """Alias for ``fit`` for backward compatibility, see there.
        """
        
        return self.fit(X, y, d=d)
    
    
    def predict_class_label(self, X):    
        """Alias for ``predict`` for backward compatibility, see there.
        """
        
        return self.predict(X)

    def predict_class_label_probability(self, X, return_std=False): 
        """Alias for ``predict_proba`` for backward compatibility, see there.
        """
        
        return self.predict_proba(X, return_std=return_std)
    
    def inflate(self, d, tau=None):    
        """Alias for ``transform`` with ``method='reference'`` for backward 
        compatibility, see there.
        """
        
        return self.transform(d, tau=None, method='reference')
    
    def compress(self, s, tau):    
        """Alias for ``inverse_transform`` with ``method='reference`` for 
        backward compatibility, see there.
        """
        
        return self.inverse_transform(s, tau, method='reference')
