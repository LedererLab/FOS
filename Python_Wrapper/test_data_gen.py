# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:37:57 2017

@author: Nehemy Lim
"""

import numpy as np
from scipy import linalg

def generate_data(nb_obs, nb_var, nb_active, cor, signal_noise_ratio):
    """
    DESCRIPTION: Generate data from a linear regression model of the form :
                  y = X*beta + noise

    USAGE:
    X, y, beta = generate_data(nb_obs, nb_var, nb_active, cor,
    signal_noise_ratio)

    INPUT ARGUMENTS:
    
    nb_obs        Number of observations, positive integer
    nb_var        Number of variables, positive integer
    nb_active     Number of active variables, positive integer <= nb_var
    cor           Magnitude of mutual correlations, scalar with 0 <= cor < 1
    signal_noise_ratio  Signal to noise ratio, positive scalar             
    
    
    OUTPUT ARGUMENTS:
    X           Input matrix, of dimension nb_obs x nb_var; each row
                corresponds to an observation.
    y           Response variable, vector of dimension nb_obs.
    beta        Regression vector of dimension nb_var.
    """
    # Generation of the design matrix X
    # rows are iid from a normal distribution with equicorrelated design
    Sigma_X = (1-cor)*np.eye(nb_var) + cor*np.ones((nb_var, nb_var))
    mu_X = np.zeros(nb_var)
    X = np.random.multivariate_normal(mu_X, Sigma_X, nb_obs)
    
    # Columns are renormalized to have Eucledian norm exactly sqrt(nb_obs)
    X = np.sqrt(nb_obs) * X / np.sqrt(np.sum(X ** 2, axis = 0))
    
    # Generation of the regression vector beta
    beta = np.zeros(nb_var)
    # nb_active variables are selected at random from the nb_var variables
    support = np.random.choice(np.arange(nb_var), nb_active, replace=False) 

    # coefficients for active variables are randomly set to -1 or 1  
    beta[support] = 2 * np.random.binomial(1, 0.5, nb_active) - 1
    
    # Rescale regression vector to given signal to noise ratio
    # such that ||X*beta||^2_2/nb_obs = signal_noise_ratio

    beta = np.sqrt( signal_noise_ratio * nb_obs 
    / (linalg.norm(X.dot(beta), 2) ** 2) ) * beta
    
    # Generation of the noise vector
    # iid entries from a standard normal distribution
    noise = np.random.randn(nb_obs)
    
    # Generation of the output vector y
    y = X.dot(beta) + noise
    
    return X, y, beta
