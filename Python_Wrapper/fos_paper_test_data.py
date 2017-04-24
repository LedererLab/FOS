#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
from scipy import spatial
import time

import hdim

def X_FOS_support( X, Y ):
	fos = hdim.X_FOS_d()

	fos( X, Y )
	return fos.ReturnSupport()

def FOS_support( X, Y ):
	fos = hdim.FOS_d( X, Y)

	fos.Algorithm()
	return fos.ReturnSupport()

def main():

	total_run_time = 0.0
	# lung cancer data: gene expressions for 500 patients (n=500, p=1000)
	cancer_data = sio.loadmat('data_obs.mat')
	X = cancer_data['data_obs']
	n,p = X.shape

	# ground truth for the interaction network for the lung cancer data
	cancer_network = sio.loadmat('GS.mat')
	ground_truth = cancer_network['DAG']


	idx_col = np.arange(p)
	C = np.zeros((p, p))

	for var_curr in idx_col:

		print( "On iteration " + str( var_curr ) + " out of " + str( len(idx_col) ) )

		y_var_curr = X[:,var_curr] # var_curr is the response to predict
		# use remaining variables as predictors
		X_minus_var_curr = X[:, idx_col[idx_col != var_curr]]

		t_start = time.time()
		# Run FOS for X=X_minus_var_curr and y=y_var_curr
		# get the corresponding support (boolean vector of length p-1)
		support_var_curr = FOS_support( X_minus_var_curr, y_var_curr )

		t_end = time.time()

		t_n = t_end - t_start

		print( "FOS took " + str( t_n ) + " seconds this iteration." )

		total_run_time += t_n

		# fill the var_curr-th column with support_var_curr
		C[idx_col[idx_col != var_curr], var_curr] = support_var_curr.flatten()


	print( "Total execution time was " + str( total_run_time ) + " seconds." )
	# construct the estimated interaction network
	E_or = np.logical_or(C, C.T)
	# Compute hamming distance between ground_truth and E_or
	hd = spatial.distance.hamming(ground_truth[np.tril_indices(p, -1)], E_or[np.tril_indices(p, -1)])

	print( hd )


if __name__ == "__main__":
    main()
