#!/usr/bin/env python3

import hdim
import test_data_gen
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio

def test_FOS( X, Y ):
	fos_test = hdim.FOS_d( X, Y )

	fos_test.Algorithm()
	return fos_test.ReturnCoefficients()

def test_X_FOS( X, Y ):
	fos_test = hdim.X_FOS_d()

	fos_test( X, Y )
	return fos_test.ReturnCoefficients()

def main():
	N = 200
	P = 500

	#ind = np.arange(P)  # the x locations for the groups

	# X, y, beta = test_data_gen.generate_data( N, P, int( math.ceil(P/10) ), 0.3, 5 )

	old_data = sio.loadmat('Bundled_data.mat')
	X = old_data['X']
	y = np.transpose( old_data['y'] )
	beta = old_data['beta']

	# Mat_dict = {}
	# Mat_dict['X'] = X
	# Mat_dict['y'] = y
	# Mat_dict['beta'] = beta

	# sio.savemat( 'Bundled_data.mat', Mat_dict )

	fos_results = test_FOS( X, y )
	x_fos_results = test_X_FOS( X, y )

	L2_sqr_norm = np.linalg.norm( x_fos_results, ord='fro' )**2

	ind = np.arange( len( fos_results ) )

	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, fos_results, width, color='#800080')
	rects2 = ax.bar(ind + width, x_fos_results, width, color='y')

	ax.legend((rects1[0], rects2[0]), ('FOS', 'X_FOS'))

	plt.show()

if __name__ == "__main__":
    main()
